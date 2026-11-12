"""Fair, explicitly labelled offline-table replay; no table outcomes reach policies."""
import copy
import math
from pathlib import Path
import time
import numpy as np
from .data import write_json, write_jsonl
from .search import SearchPolicy, initial_design, measured_incumbent
from .surrogate import NeuralSurrogate, observation_key


def synthetic_table():
    """Deterministic analytic fixture, not a simulation validated against vLLM."""
    from .experiment import candidates_for
    study = {'space': {'max_num_batched_tokens': [256, 512, 1024], 'max_num_seqs': [2, 4, 8], 'gpu_memory_utilization': [.5, .6]},
             'fixed_config': {'precision': 'bf16', 'tensor_parallel_size': 1}}
    rows = []; rng = np.random.default_rng(724)
    for label, load, length in [('low', .05, 100), ('mid', .1, 100), ('high', .2, 100), ('unseen-mix', .15, 350)]:
        workload = {'scenario_id': label, 'profile': 'mixed' if label == 'unseen-mix' else 'short', 'arrival_rate': load,
                    'input_tokens_mean': length, 'input_tokens_p95': length * 1.3, 'output_token_limit': 1024, 'burstiness': 1.0}
        for i, c in enumerate(candidates_for(study, workload)):
            config = c['config']; capacity = .08 + config['max_num_seqs'] * .018 + math.log2(config['max_num_batched_tokens'] / 256 + 1) * .015 - (length - 100) / 20000
            latency = 8 + 40 * load / capacity + config['max_num_seqs'] * .15
            for rep in range(2):
                failed = load > capacity * 1.1; goodput = 0 if failed else min(load, capacity) * .72 * (1 + rng.normal(0, .03))
                rows.append({**c, 'is_default': i == 0, 'run_id': f'synthetic-{label}-{i}-{rep}', 'repetition': rep, 'source': 'synthetic',
                             'partition': 'holdout' if label == 'unseen-mix' else 'train', 'environment_hash': 'SYNTHETIC_NO_HARDWARE',
                             'status': 'timeout' if failed else 'ok', 'metrics': {'quality_goodput': goodput, 'p99_e2e_s': None if failed else latency * (1 + rng.normal(0, .03)),
                             'p99_e2e_censored': failed, 'accuracy': 0 if failed else .72, 'feasible': not failed, 'offered': 100, 'completed': 0 if failed else 100, 'peak_memory_bytes': None},
                             'cost': {'accelerator_seconds': 90 + 5 * rep, 'wall_seconds': 90 + 5 * rep, 'optimizer_cpu_seconds': 0}})
    return rows


def _evidence_source(table):
    if not table:
        raise ValueError('empty observation table')
    sources = {r['source'] for r in table}
    if len(sources) != 1 or not sources <= {'measured', 'synthetic'}:
        raise ValueError('never mix measured and synthetic evidence; source must be explicit')
    environments = {r.get('environment_hash') for r in table}
    if len(environments) != 1 or None in environments:
        raise ValueError('one recorded environment is required; no silent hardware transfer')
    return next(iter(sources))


def replay(table, output_dir, *, budget_s=1000, maximum_trial_cost_s=100, initial_points=6,
           seeds=(11, 29, 47), scenario='mid', neural_epochs=100, repetition=0):
    """Replay a predeclared repetition, never cherry-picking by observed outcome.

    A maximum-cost gate is deliberately conservative: it cannot peek at the next
    unobserved trial's cost to fit an additional cheap candidate into the budget.
    The fixed default control gets one trial; it is not a fourth searching method.
    Replay CPU fitting occurs without reserved inference accelerators. Synthetic
    table costs are prescribed fixture values, not measured accelerator-time.
    """
    source = _evidence_source(table)
    if not math.isfinite(budget_s) or budget_s < 0 or not math.isfinite(maximum_trial_cost_s) or maximum_trial_cost_s <= 0:
        raise ValueError('finite nonnegative budget and positive maximum trial cost required')
    if initial_points < 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('nonnegative initial_points and distinct nonempty seeds required')
    rows = [r for r in table if r['scenario_id'] == scenario and r.get('partition') not in {'test', 'holdout'}]
    by_id = {}
    for row in rows:
        if row['repetition'] != repetition:
            continue
        if row['config_id'] in by_id:
            raise ValueError('duplicate configuration/scenario/repetition in replay table')
        cost = row['cost']['accelerator_seconds']
        if not math.isfinite(cost) or cost < 0 or cost > maximum_trial_cost_s:
            raise ValueError('predeclared maximum trial cost does not bound nonnegative table costs')
        by_id[row['config_id']] = row
    if not by_id:
        raise ValueError('no development candidates for the declared repetition')
    if set(by_id) != {r['config_id'] for r in rows}:
        raise ValueError('declared repetition is missing for some development configurations')
    # Sorting makes a shuffled raw table yield the same seeded experiment.
    candidates = [{k: copy.deepcopy(v) for k, v in by_id[cid].items()
                   if k in {'config_id', 'scenario_id', 'config', 'workload', 'is_default', 'known_valid'}}
                  for cid in sorted(by_id)]
    p = Path(output_dir)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError('replay output is nonempty; preserve it and use a fresh run directory')
    p.mkdir(parents=True, exist_ok=True)
    curves, summaries = [], []
    for seed in seeds:
        common = initial_design(candidates, initial_points, seed)
        for method in ('default', 'random', 'gp', 'neural'):
            policy = SearchPolicy(method, seed, neural_epochs=neural_epochs)
            observed, cost, cpu, steps = [], 0, 0, 0
            while cost + maximum_trial_cost_s <= budget_s:
                shared_start = method != 'default' and len(observed) < len(common)
                if shared_start:
                    candidate = common[len(observed)]
                    decision = {'method_used': 'shared_initial_design', 'optimizer_cpu_seconds': 0.0}
                else:
                    candidate = policy.choose(candidates, observed)
                    decision = copy.deepcopy(policy.last_decision)
                if candidate is None:
                    break
                # This is the only line that reveals an unobserved outcome.
                row = copy.deepcopy(by_id[candidate['config_id']])
                observed.append(row)
                cost += row['cost']['accelerator_seconds']; cpu += decision.get('optimizer_cpu_seconds', 0); steps += 1
                best = measured_incumbent(observed)
                curves.append({'source': source, 'mode': 'offline_replay', 'method': method, 'seed': seed, 'trial': steps,
                               'cumulative_accelerator_seconds': cost, 'optimizer_cpu_seconds': cpu,
                               'best_feasible_goodput': best['quality_goodput'] if best else 0.0, 'selected_config_id': row['config_id'],
                               'selected_run_id': row['run_id'], 'acquisition': decision})
            policy.save(p / f'{method}-seed{seed}.json')
            summaries.append({'source': source, 'mode': 'offline_replay', 'method': method, 'seed': seed, 'trials': steps,
                              'cost_s': cost, 'optimizer_cpu_seconds': cpu, 'best': measured_incumbent(observed),
                              'initial_config_ids': [c['config_id'] for c in common] if method != 'default' else [],
                              'repetition': repetition, 'cost_interpretation': 'prescribed synthetic fixture' if source == 'synthetic' else 'replayed measured reservation cost'})
    write_jsonl(p / 'curves.jsonl', curves)
    write_json(p / 'summary.json', summaries, immutable=True)
    return curves, summaries


def train_diagnostic(table, output_dir, epochs=150):
    """Separate full-table surrogate diagnostic; not free observations for replay."""
    source = _evidence_source(table)
    if any(r['partition'] not in {'train', 'validation', 'holdout'} for r in table):
        raise ValueError('diagnostic accepts only declared train/validation/holdout partitions')
    train = [r for r in table if r['partition'] in {'train', 'validation'}]
    hold = [r for r in table if r['partition'] == 'holdout']
    if not train or not hold:
        raise ValueError('diagnostic needs both development and held-out workload records')
    if {r['scenario_id'] for r in train} & {r['scenario_id'] for r in hold}:
        raise ValueError('unseen-workload diagnostic requires disjoint scenario IDs')
    p = Path(output_dir)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError('diagnostic output is nonempty; preserve it and use a fresh run directory')
    p.mkdir(parents=True, exist_ok=True)
    surrogate = NeuralSurrogate(max_epochs=epochs, seed=724)
    start, cpu_start = time.perf_counter(), time.process_time()
    report = surrogate.fit(train)
    predictions = surrogate.predict(hold); surrogate.save(p / 'surrogate.pt')
    pairs = [{'source': r['source'], 'run_id': r['run_id'], 'scenario_id': r['scenario_id'],
              'actual_goodput': r['metrics']['quality_goodput'], 'actual_p99': r['metrics']['p99_e2e_s'], **pred}
             for r, pred in zip(hold, predictions)]
    training_keys = set(report['training_groups'])
    constant = float(np.mean([r['metrics']['quality_goodput'] for r in train if observation_key(r) in training_keys]))
    latency_pairs = [r for r in pairs if r['actual_p99'] is not None]
    write_jsonl(p / 'heldout-predictions.jsonl', pairs)
    write_json(p / 'training.json', {
        'source': source, 'cpu_training_wall_s': time.perf_counter() - start, 'optimizer_cpu_seconds': time.process_time() - cpu_start,
        'training_report': report, 'heldout_goodput_mae': float(np.mean([abs(r['actual_goodput'] - r['goodput_mean']) for r in pairs])),
        'training_mean_baseline_goodput_mae': float(np.mean([abs(r['actual_goodput'] - constant) for r in pairs])),
        'heldout_p99_mae_s': float(np.mean([abs(r['actual_p99'] - r['p99_mean']) for r in latency_pairs])) if latency_pairs else None,
        'heldout_uncensored_latency_records': len(latency_pairs),
        'interpretation': 'software validation only; separate from optimizer observations' if source == 'synthetic' else 'unseen declared workload prediction; no hardware transfer claim'
    }, immutable=True)
    return pairs
