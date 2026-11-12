#!/usr/bin/env python3
"""Fit a deliberately non-deployable diagnostic to the three measured pilot rows.

No server starts, no official test questions are read, no raw observation is
edited, and no prediction is admitted to the configuration search. A fitted
checkpoint demonstrates gradient-based learning only, not useful prediction.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vllm_tuner.data import read_jsonl, write_json, write_jsonl
from vllm_tuner.surrogate import NeuralSurrogate, grouped_split, targets

RUN_IDS = ('pilot-metal-default-001', 'screen-metal-00', 'allowance-metal-2048-00')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_excerpt(path, first, last):
    lines = path.read_text().splitlines()
    return {'file': str(path), 'sha256': sha(path), 'first_line': first, 'last_line': last,
            'text': '\n'.join(lines[first - 1:last])}


def resolve_defaults(raw_root, runtime_packages=None, prior_evidence=None):
    log = raw_root / 'runs' / RUN_IDS[0] / 'server.log'
    if prior_evidence:
        evidence = json.loads(Path(prior_evidence).read_text())
        if evidence['default_server_log_sha256'] != sha(log):
            raise ValueError('saved default-resolution evidence does not match the immutable pilot log')
        if evidence['resolved_values'] != {'max_num_batched_tokens': 2048, 'max_num_seqs': 256}:
            raise ValueError('unexpected default resolution')
        return evidence
    if runtime_packages is None:
        raise ValueError('provide installed runtime site-packages or saved default-resolution evidence')
    packages = Path(runtime_packages)
    args = packages / 'vllm/engine/arg_utils.py'
    platform = packages / 'vllm_metal/platform.py'
    interface = packages / 'vllm/platforms/interface.py'
    vconfig = packages / 'vllm/config/vllm.py'
    if 'max_num_batched_tokens=2048' not in log.read_text():
        raise ValueError('default scheduler token limit not confirmed by actual server log')
    checks = [
        ('UsageContext.OPENAI_API_SERVER: 2048,', args),
        ('UsageContext.OPENAI_API_SERVER: 256,', args),
        ('_enum: PlatformEnum = PlatformEnum.OOT', platform),
        ('return self._enum == PlatformEnum.CPU', interface),
        ('performance_mode: PerformanceMode = "balanced"', vconfig),
    ]
    for expected, file in checks:
        if expected not in file.read_text():
            raise ValueError(f'installed default-resolution source changed: {file}')
    versions = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in importlib.metadata.distributions(path=[str(packages)])}
    return {
        'resolved_values': {'max_num_batched_tokens': 2048, 'max_num_seqs': 256},
        'vllm_version': versions['vllm'], 'vllm_metal_version': versions['vllm-metal'],
        'default_server_log': str(log), 'default_server_log_sha256': sha(log),
        'resolution_basis': {
            'max_num_batched_tokens': 'Actual Metal engine startup log emits 2048.',
            'max_num_seqs': 'Installed vLLM source inference: 16GB device takes ordinary-memory OPENAI_API_SERVER default256; Metal PlatformEnum.OOT is not PlatformEnum.CPU; balanced mode does not double it; min(256,2048)=256; no max_num_seqs override in logged non-default CLI arguments.',
        },
        'source_evidence': [source_excerpt(args, 2920, 2953), source_excerpt(args, 3110, 3134),
                            source_excerpt(args, 3178, 3186), source_excerpt(platform, 99, 106),
                            source_excerpt(interface, 208, 209), source_excerpt(vconfig, 430, 435)],
        'caveat': 'Sequence limit is inferred from installed initialization code, not separately emitted by this startup log. Original null launch overrides remain untouched.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/measured/observations.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('results/measured/pilot-model'))
    parser.add_argument('--runtime-site-packages', type=Path)
    parser.add_argument('--resolution-evidence', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('preserve existing diagnostic output; choose a fresh output directory')
    original_hash = sha(args.input)
    raw = read_jsonl(args.input)
    by_id = {r['run_id']: r for r in raw}
    if set(by_id) != set(RUN_IDS) or len(raw) != 3:
        raise ValueError('this preregistered diagnostic is restricted to the three named pilot runs')
    if {r['source'] for r in raw} != {'measured'} or any(r['partition'] != 'train' for r in raw):
        raise ValueError('only measured pilot development records are permitted')
    if len({r['environment_hash'] for r in raw}) != 1:
        raise ValueError('pilot environment mismatch')
    if any(r['metrics']['feasible'] for r in raw):
        raise ValueError('diagnostic assumptions changed: a feasible label is present')
    if by_id[RUN_IDS[1]]['trace_hash'] != by_id[RUN_IDS[2]]['trace_hash']:
        raise ValueError('allowance runs must stay together as one shared-trace group')
    if by_id[RUN_IDS[0]]['trace_hash'] == by_id[RUN_IDS[1]]['trace_hash']:
        raise ValueError('only one independent trace group; do not train this diagnostic')
    evidence = resolve_defaults(args.input.parent, args.runtime_site_packages, args.resolution_evidence)
    derived = [copy.deepcopy(by_id[run_id]) for run_id in RUN_IDS]
    for row in derived:
        row['diagnostic_only'] = True
        row['original_partition'] = row['partition']
        row['original_config'] = copy.deepcopy(row['config'])
        row['partition'] = 'validation' if row['run_id'] == RUN_IDS[0] else 'train'
        if row['run_id'] == RUN_IDS[0]:
            for name, value in evidence['resolved_values'].items():
                if row['config'][name] is not None:
                    raise ValueError('expected omitted default launch overrides')
                row['config'][name] = value
            row['feature_resolution'] = 'default-resolution.json; original launch config retained in original_config'
    train, validation = grouped_split(derived, seed=724)
    train_target, train_mask, train_feasibility = targets(train)
    val_target, val_mask, val_feasibility = targets(validation)
    if len(train) != 2 or len(validation) != 1 or int(train_mask[:, 0].sum()) != 2:
        raise ValueError('unexpected grouped pilot partition; do not train')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'default-resolution.json', evidence, immutable=True)
    write_jsonl(args.output / 'derived-inputs.jsonl', derived)
    torch.set_num_threads(1)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    model = NeuralSurrogate(seed=724, ensemble_size=3, max_epochs=args.epochs, patience=20, standardize_targets=False)
    training = model.fit(derived)
    predictions = model.predict(validation)
    # Save a checkpoint explicitly marked in its own report, not only its filename.
    model.report.update({
        'diagnostic_fit_only': True, 'reliable_for_search': False,
        'independent_trace_groups_total': 2, 'independent_trace_groups_train': 1,
        'independent_trace_groups_validation': 1,
        'head_reliability': {'goodput': False, 'p99_e2e': False, 'feasibility': False},
        'all_observed_feasibility_labels_negative': True,
        'reason': 'Three pilot rows, two independent trace groups, no positive feasibility label and no validation latency target; protocols differ. No optimizer comparison or held-out test evaluation.',
    })
    model.save(args.output / 'diagnostic-ensemble.pt')
    elapsed_wall, elapsed_cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    reloaded = NeuralSurrogate.load(args.output / 'diagnostic-ensemble.pt')
    if reloaded.predict(validation) != predictions:
        raise RuntimeError('checkpoint reload prediction mismatch')
    actual = np.asarray([r['metrics']['quality_goodput'] for r in validation])
    predicted = np.asarray([p['goodput_mean'] for p in predictions])
    constant = float(np.mean([r['metrics']['quality_goodput'] for r in train]))
    rows = []
    for row, pred in zip(validation, predictions):
        rows.append({'run_id': row['run_id'], 'population': 'one internal group validation pilot; NOT official test',
                     'actual_goodput': row['metrics']['quality_goodput'], 'actual_p99_e2e_s': None,
                     'actual_feasible': False, 'prediction_reliable': False,
                     'unvalidated_prediction': pred, 'constant_training_mean_goodput': constant})
    write_jsonl(args.output / 'internal-validation-predictions.jsonl', rows)
    summary = {
        'status': 'diagnostic_fit_only', 'source': 'measured', 'checkpoint_trained': True,
        'original_journal_sha256_before': original_hash, 'original_journal_sha256_after': sha(args.input),
        'total_records': 3, 'independent_trace_groups_total': 2,
        'training_records': len(train), 'training_independent_trace_groups': 1,
        'internal_validation_records': len(validation), 'validation_independent_trace_groups': 1,
        'train_run_ids': [r['run_id'] for r in train], 'validation_run_ids': [r['run_id'] for r in validation],
        'training_target_counts': {'goodput': int(train_mask[:, 0].sum()), 'uncensored_p99': int(train_mask[:, 1].sum()),
                                   'positive_feasibility': int(train_feasibility.sum()), 'negative_feasibility': len(train)},
        'validation_target_counts': {'goodput': int(val_mask[:, 0].sum()), 'uncensored_p99': int(val_mask[:, 1].sum()),
                                     'positive_feasibility': int(val_feasibility.sum()), 'negative_feasibility': len(validation)},
        'internal_validation_goodput_mae': float(np.mean(np.abs(actual - predicted))),
        'training_mean_constant_goodput': constant,
        'training_mean_constant_validation_mae': float(np.mean(np.abs(actual - constant))),
        'always_zero_goodput_validation_mae': float(np.mean(np.abs(actual))),
        'feasibility_constant_negative_brier': 0.0,
        'neural_feasibility_brier': float(np.mean([p['feasible_probability'] ** 2 for p in predictions])),
        'p99_validation_mae_s': None, 'p99_validation_reason': 'No uncensored validation target; the one training P99 is based on only two completions.',
        'parameter_count_per_member': sum(p.numel() for p in model.models[0].parameters()),
        'ensemble_members': 3, 'optimizer_cpu_seconds': elapsed_cpu, 'optimizer_wall_seconds': elapsed_wall,
        'additional_reserved_accelerator_seconds': 0.0,
        'reliable_for_search': False, 'quality_preservation_proven': False, 'search_comparison_run': False,
        'official_test_used': False, 'head_reliability': model.report['head_reliability'],
        'split_rule': 'Earlier default pilot is internal validation. Both later allowance rows remain together in training because their trace hash is identical. Split fixed before training; no result-based split selection.',
        'limitations': ['One independent training group cannot establish a response surface.',
                        'All three feasibility labels are negative; feasibility discrimination cannot be assessed.',
                        'Only one uncensored P99 training target and no validation P99; no tail prediction evidence.',
                        'Default protocol differs in arrival rate/deadline/measurement duration, so observations are not a controlled serving comparison.',
                        'Early stopping on one group is unstable; ensemble spread is not a calibrated confidence interval.',
                        'Any apparent validation advantage over a constant is uninformative with one all-zero validation outcome.'],
        'training_report': model.report,
    }
    if summary['original_journal_sha256_after'] != original_hash:
        raise RuntimeError('raw journal changed during diagnostic')
    write_json(args.output / 'diagnostic-summary.json', summary, immutable=True)
    write_json(args.output / 'provenance.json', {
        'script_sha256': sha(__file__), 'input_journal': str(args.input), 'input_sha256': original_hash,
        'derived_inputs_sha256': sha(args.output / 'derived-inputs.jsonl'),
        'checkpoint_sha256': sha(args.output / 'diagnostic-ensemble.pt'),
        'default_resolution_sha256': sha(args.output / 'default-resolution.json'),
        'epochs_limit': args.epochs, 'seed': 724, 'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'runtime': 'CPU only; all vLLM servers stopped before fitting',
    }, immutable=True)
    print(json.dumps({k: summary[k] for k in ('status', 'training_records', 'internal_validation_records',
                      'training_target_counts', 'validation_target_counts', 'internal_validation_goodput_mae',
                      'training_mean_constant_validation_mae', 'always_zero_goodput_validation_mae',
                      'optimizer_cpu_seconds', 'additional_reserved_accelerator_seconds', 'reliable_for_search')}, indent=2))


if __name__ == '__main__':
    main()
