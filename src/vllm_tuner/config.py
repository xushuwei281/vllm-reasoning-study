"""Conservative serving-space validation and release-specific CLI verification.

Known-valid configuration is necessary, never sufficient: a live smoke run must
still establish memory feasibility and correct backend execution.
"""
from __future__ import annotations
import importlib.metadata
import math
import re
import shutil
import subprocess
from typing import Any, Mapping

MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MODEL = {'name': 'Qwen/Qwen3-1.7B', 'revision': MODEL_REVISION,
         'tokenizer_revision': MODEL_REVISION, 'license': 'Apache-2.0',
         'attention_heads': 16, 'kv_heads': 8, 'max_model_len': 4096,
         'native_max_model_len': 40960}
BASE_CONFIG = {'precision': 'bf16', 'max_num_batched_tokens': 512,
               'max_num_seqs': 8, 'gpu_memory_utilization': 0.55,
               'tensor_parallel_size': 1, 'chunked_prefill': True,
               'prefix_caching': False, 'kv_cache_dtype': 'auto'}


def validate_config(config: Mapping[str, Any], capabilities: Mapping[str, Any],
                    workload: Mapping[str, Any] | None = None) -> list[str]:
    """Return every known invalidity; capability claims require hardware preflight.

    capabilities: device_count, allowed_tp_sizes, supported_precisions,
    chunked_prefill, prefix_caching, max_model_len, attention_heads, kv_heads,
    max_memory_fraction; optional quantization_evidence dict.
    """
    errors: list[str] = []
    tp = config.get('tensor_parallel_size', 1)
    devices = capabilities.get('device_count', 0)
    if not isinstance(tp, int) or isinstance(tp, bool) or tp < 1:
        errors.append('tensor_parallel_size must be a positive integer')
    else:
        if tp > devices:
            errors.append('tensor parallel size exceeds the fixed available fleet')
        if tp not in capabilities.get('allowed_tp_sizes', [1]):
            errors.append('tensor parallel size lacks a verified backend/topology capability')
        heads = capabilities.get('attention_heads', 16)
        kv_heads = capabilities.get('kv_heads', 8)
        if heads % tp or (kv_heads % tp and tp % kv_heads):
            errors.append('tensor/head partition is invalid for this model')
    precision = config.get('precision', 'bf16')
    if precision not in capabilities.get('supported_precisions', []):
        errors.append('precision has no verified hardware/backend support')
    if precision not in ('bf16', 'fp16'):
        evidence = capabilities.get('quantization_evidence') or {}
        required = {'scheme', 'checkpoint_revision', 'calibration_split', 'scales',
                    'excluded_layers', 'weight_dtype', 'activation_dtype',
                    'kernel_verification', 'development_accuracy'}
        if not required.issubset(evidence):
            errors.append('quantization requires a complete kernel/calibration/quality evidence record')
        if evidence.get('calibration_split') not in ('train', 'development'):
            errors.append('quantization calibration may not use official test data')
    for key in ('max_num_batched_tokens', 'max_num_seqs'):
        value = config.get(key)
        # None means use actual version/hardware-dependent engine default.
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            errors.append(f'{key} must be a positive integer or None for engine default')
    batch, seqs = config.get('max_num_batched_tokens'), config.get('max_num_seqs')
    if isinstance(batch, int) and isinstance(seqs, int) and batch < seqs:
        errors.append('batched token limit must accommodate at least one token per sequence')
    memory = config.get('gpu_memory_utilization', 0.55)
    if isinstance(memory, bool) or not isinstance(memory, (float, int)) or not math.isfinite(memory) or not 0 < memory <= 1:
        errors.append('gpu_memory_utilization must lie in (0, 1]')
    elif memory > capabilities.get('max_memory_fraction', 0.90):
        errors.append('memory fraction exceeds preregistered headroom limit')
    chunked = config.get('chunked_prefill', True)
    if chunked and not capabilities.get('chunked_prefill', False):
        errors.append('chunked prefill not verified')
    if config.get('prefix_caching', False) and not capabilities.get('prefix_caching', False):
        errors.append('prefix caching not verified')
    model_len = capabilities.get('max_model_len', 4096)
    if not chunked and isinstance(batch, int) and batch < model_len:
        errors.append('non-chunked prefill token budget must cover max_model_len')
    if config.get('kv_cache_dtype', 'auto') != 'auto':
        errors.append('this bounded study fixes KV cache dtype to auto/reference model dtype')
    if config.get('placement', 'colocated') != 'colocated' or config.get('kv_connector'):
        errors.append('disaggregation is a separate unimplemented capability-gated extension')
    if config.get('hccl_buffsize') is not None or config.get('hccl'):
        errors.append('HCCL controls are not enabled in the verified local search space')
    if workload:
        limit = workload.get('output_token_limit', 0)
        largest_input = workload.get('input_tokens_max', workload.get('input_tokens_p95', 0))
        if (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
                or not isinstance(largest_input, (int, float)) or not math.isfinite(largest_input)
                or largest_input < 0 or largest_input + limit > model_len):
            errors.append('input length plus output allowance exceeds the fixed context length or is invalid')
    return errors


def build_serve_command(config: Mapping[str, Any], model: Mapping[str, Any],
                        port: int = 8000, host: str = '127.0.0.1',
                        executable: str = 'vllm') -> list[str]:
    """Build argv, never a shell string. Flags must pass preflight on deployment.

    Reasoning parser deliberately omitted: retain raw <think> blocks for exact
    token accounting in the evaluator, with thinking enabled in chat template.
    """
    for key in ('revision', 'tokenizer_revision'):
        if not re.fullmatch(r'[0-9a-f]{40}', str(model.get(key, ''))):
            raise ValueError(f'{key} must be an immutable 40-character commit SHA')
    precision = config.get('precision', 'bf16')
    if precision not in ('bf16', 'fp16'):
        raise ValueError('quantization requires a backend-specific implementation; no universal flag')
    command = [executable, 'serve', model['name'], '--served-model-name', model['name'],
               '--revision', model['revision'],
               '--tokenizer-revision', model['tokenizer_revision'], '--host', host,
               '--port', str(port), '--dtype', {'bf16': 'bfloat16', 'fp16': 'float16'}[precision],
               '--max-model-len', str(model['max_model_len']),
               '--tensor-parallel-size', str(config.get('tensor_parallel_size', 1)),
               '--gpu-memory-utilization', str(config.get('gpu_memory_utilization', 0.55)),
               '--kv-cache-dtype', 'auto', '--generation-config', 'vllm', '--seed', '0',
               '--enable-chunked-prefill' if config.get('chunked_prefill', True) else '--no-enable-chunked-prefill',
               '--enable-prefix-caching' if config.get('prefix_caching', False) else '--no-enable-prefix-caching']
    for key in ('max_num_batched_tokens', 'max_num_seqs'):
        if config.get(key) is not None:
            command += ['--' + key.replace('_', '-'), str(config[key])]
    return command


def preflight_runtime(expected_version: str = '0.32.0', executable: str = 'vllm',
                      output_dir: str | None = None) -> dict[str, Any]:
    """Fail closed unless release and every emitted CLI flag exist locally.

    Saves exact CLI help. Does not download weights, start a server, or infer
    kernel compatibility from flag presence. Core '+cpu' wheel suffix is valid
    for Metal's external device plugin.
    """
    from pathlib import Path
    result: dict[str, Any] = {'expected_version': expected_version, 'ok': False,
                             'errors': [], 'serve_help': '', 'bench_help': ''}
    resolved = shutil.which(executable)
    if not resolved:
        result['errors'].append(f'{executable} executable not found')
        return result
    result['executable'] = resolved
    try:
        version = subprocess.run([resolved, '--version'], capture_output=True, text=True,
                                 timeout=90, check=False)
        result['version_output'] = version.stdout + version.stderr
        versions = re.findall(r'(?<![\d.])(\d+\.\d+\.\d+(?:\+[\w.]+)?)', version.stdout)
        if version.returncode != 0 or not any(v.split('+')[0] == expected_version for v in versions):
            result['errors'].append('installed vLLM does not match the declared core release')
        # v0.10.2 treats --help=<word> as a substring search; "all" matches
        # parallel/allowed flags, not the full CLI. Its plain argparse help is
        # complete. Newer pinned 0.32.0 keeps its explicit full-help spelling.
        serve_help_flag = '--help' if expected_version == '0.10.2' else '--help=all'
        result['help_commands'] = {'serve': [resolved, 'serve', serve_help_flag],
                                   'bench': [resolved, 'bench', 'serve', '--help']}
        for key, args in [('serve_help', ['serve', serve_help_flag]),
                          ('bench_help', ['bench', 'serve', '--help'])]:
            proc = subprocess.run([resolved, *args], capture_output=True, text=True,
                                  timeout=90, check=False)
            result[key] = proc.stdout + proc.stderr
            if proc.returncode:
                result['errors'].append(f'{key} exited {proc.returncode}')
        flags = {x for x in build_serve_command(BASE_CONFIG, MODEL) if x.startswith('--')}
        for flag in sorted(flags):
            if flag not in result['serve_help']:
                result['errors'].append(f'CLI flag missing from installed serve help: {flag}')
    except (OSError, subprocess.TimeoutExpired) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['ok'] = not result['errors']
    if output_dir:
        root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
        for key in ('serve_help', 'bench_help'):
            (root / f'{key}.txt').write_text(result[key])
    return result
