"""Sanitized, reproducible runtime/environment inventory (no serials or IPs)."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _command(args: list[str]) -> dict[str, Any]:
    if not shutil.which(args[0]):
        return {'available': False, 'output': None, 'returncode': None}
    try:
        proc = subprocess.run(args, text=True, capture_output=True, timeout=30)
        return {'available': True, 'output': proc.stdout.strip(), 'returncode': proc.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': True, 'output': None, 'error': type(exc).__name__}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_hash(manifest: dict) -> str:
    stable = {k: v for k, v in manifest.items() if k not in ('collected_at_utc', 'environment_hash')}
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def collect_environment() -> dict[str, Any]:
    packages = ['torch', 'vllm', 'vllm-metal', 'mlx', 'mlx-lm', 'vllm-ascend', 'torch-npu',
                'transformers', 'huggingface-hub', 'llmcompressor', 'compressed-tensors',
                'numpy', 'scipy', 'scikit-learn', 'httpx', 'aiohttp', 'pytest', 'matplotlib']
    result: dict[str, Any] = {
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'os': {'system': platform.system(), 'release': platform.release(),
               'macos': platform.mac_ver()[0] or None, 'architecture': platform.machine()},
        'python': platform.python_version(),
        'cpu': {'logical_count': os.cpu_count(), 'name': platform.processor() or None},
        'host_ram_bytes': None,
        'packages': {p: _version(p) for p in packages},
        'accelerators': [], 'topology': 'not detected',
        'communication': {'NCCL': None, 'HCCL': None, 'CANN': None},
        'drivers': {},
        'notes': ['GPU memory is shared host unified memory on Apple Silicon; do not add it to host RAM.']}
    if platform.system() == 'Darwin':
        memory = _command(['sysctl', '-n', 'hw.memsize'])
        result['host_ram_bytes'] = int(memory['output']) if memory.get('output', '').isdigit() else None
        info = _command(['system_profiler', '-json', 'SPHardwareDataType', 'SPDisplaysDataType'])
        if info.get('output'):
            try:
                data = json.loads(info['output'])
                hardware = data.get('SPHardwareDataType', [{}])[0]
                result['cpu']['name'] = hardware.get('chip_type')
                result['cpu']['model'] = hardware.get('machine_model')
                for item in data.get('SPDisplaysDataType', []):
                    result['accelerators'].append({'type': 'apple_metal',
                        'name': item.get('sppci_model', item.get('_name')),
                        'cores': item.get('sppci_cores'), 'memory_kind': 'shared_unified',
                        'shared_memory_bytes': result['host_ram_bytes'],
                        'metal_support': item.get('spdisplays_metal')})
                result['topology'] = 'single SoC; CPU and Apple GPU share unified memory'
                result['drivers']['metal'] = 'OS bundled; see os.macos'
            except (ValueError, IndexError):
                result['notes'].append('system_profiler JSON unavailable')
    elif platform.system() == 'Linux':
        try:
            memtotal = next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
            result['host_ram_bytes'] = int(memtotal.split()[1]) * 1024
        except (OSError, StopIteration, ValueError):
            pass
    nvidia = _command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'])
    if nvidia.get('output'):
        for row in nvidia['output'].splitlines():
            name, mem, driver = [v.strip() for v in row.split(',')]
            result['accelerators'].append({'type': 'cuda', 'name': name, 'memory_bytes': int(mem) * 1024**2})
            result['drivers']['nvidia'] = driver
        result['topology'] = _command(['nvidia-smi', 'topo', '-m']).get('output')
    result['npu_smi_available'] = shutil.which('npu-smi') is not None
    if result['npu_smi_available']:
        # Raw NPU inventories may include addresses/identities. Require a manually
        # sanitized deployment manifest rather than automatically exporting them.
        result['notes'].append('Ascend detected: attach sanitized NPU count/memory/topology and CANN/HCCL versions before enabling.')
    if result['packages']['torch']:
        try:
            import torch
            result['torch_runtime'] = {'cuda': torch.version.cuda,
                'cuda_available': torch.cuda.is_available(),
                'mps_available': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())}
            if torch.cuda.is_available():
                result['communication']['NCCL'] = list(torch.cuda.nccl.version()) if isinstance(torch.cuda.nccl.version(), tuple) else torch.cuda.nccl.version()
        except Exception as exc:
            result['notes'].append(f'torch runtime inventory failed: {type(exc).__name__}')
    result['environment_hash'] = environment_hash(result)
    return result


def local_capabilities(manifest: dict) -> dict:
    """Facts to guide a pilot; CLI/kernel success must still be verified separately."""
    devices = manifest.get('accelerators', [])
    metal = any(d['type'] == 'apple_metal' for d in devices)
    cuda = [d for d in devices if d['type'] == 'cuda']
    packages = manifest.get('packages', {})
    runtime = bool(packages.get('vllm')) and (bool(cuda) or (metal and bool(packages.get('vllm-metal'))))
    return {'backend': 'metal_mlx' if metal else ('cuda' if cuda else 'unverified'),
            'runtime_installed': runtime, 'device_count': len(devices),
            'allowed_tp_sizes': [1], 'supported_precisions': ['bf16', 'fp16'] if runtime else [],
            'chunked_prefill': runtime, 'prefix_caching': runtime, 'max_model_len': 4096,
            'attention_heads': 16, 'kv_heads': 8, 'max_memory_fraction': 0.65 if metal else 0.9,
            'quantization_evidence': None, 'disaggregation': False,
            'HCCL': False, 'requires_live_smoke': True}
