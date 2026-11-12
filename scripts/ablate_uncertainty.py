"""POSTHOC OFFLINE REPLAY over a measured static table; no serving jobs.

Run on CPU after the live campaign. Example from the project directory:
  python scripts/ablate_uncertainty.py --observations results/continuation/search/*/observations.jsonl \
    --budget-ledger results/measured/budget.jsonl --study configs/study-cloud.json \
    --output results/posthoc-uncertainty-ablation.json

The global budget ledger is read only to establish chronology and measured cost.
Replayed source costs are bookkeeping for simulated choices, never new charges.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from vllm_tuner.data import digest, read_jsonl, write_json
from vllm_tuner.search import SearchPolicy, initial_design, measured_incumbent
from vllm_tuner.surrogate import connected_observation_groups, grouped_split, observation_key

MODE = 'posthoc_offline_replay_over_measured_table'
LABEL = 'POSTHOC OFFLINE REPLAY — simulated search over measured observations; no new hardware finding'


def first_chronological_table(observations, ledger_events, expected_candidates):
    """Select first live measurement per config without consulting its quality."""
    expected = {observation_key(c): c for c in expected_candidates}
    if not expected or len(expected) != len(expected_candidates):
        raise ValueError('distinct frozen candidate descriptors are required')
    if len({c['scenario_id'] for c in expected_candidates}) != 1:
        raise ValueError('one frozen workload scenario is required')
    starts, finishes = {}, {}
    for position, event in enumerate(ledger_events):
        if event.get('event') == 'begin':
            run_id = event['run_id']
            if run_id in starts:
                raise ValueError('duplicate reservation begin event')
            starts[run_id] = (position, event)
        elif event.get('event') == 'finish':
            if event['run_id'] in finishes:
                raise ValueError('duplicate reservation finish event')
            finishes[event['run_id']] = event
    by_run = {}; environments = set(); ordered = []
    for row in observations:
        if row.get('source') != 'measured':
            raise ValueError('ablation accepts measured evidence only; synthetic mixtures are forbidden')
        if row.get('partition') not in {'train', 'validation'} or row.get('workload', {}).get('partition') in {'holdout', 'test'}:
            raise ValueError('held-out outcomes cannot enter this development ablation')
        key = observation_key(row)
        if key not in expected or row['config'] != expected[key]['config']:
            raise ValueError('observation outside the frozen configuration space')
        run_id = row['run_id']
        if run_id in by_run:
            if row != by_run[run_id]:
                raise ValueError('conflicting duplicate observation')
            continue
        if run_id not in starts or run_id not in finishes:
            raise ValueError('each empirical table row needs completed live reservation provenance')
        position, start = starts[run_id]
        if start.get('phase') != 'search' or start.get('devices', 0) < 1:
            raise ValueError('only accelerator-backed live search observations belong in the table')
        stamp = datetime.fromisoformat(start['started_at_utc'])
        if stamp.tzinfo is None:
            raise ValueError('reservation chronology needs timezone-aware UTC timestamps')
        cost = row.get('cost', {}).get('accelerator_seconds')
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError('measured table costs must be finite positive reservations')
        if not math.isclose(cost, finishes[run_id]['accelerator_seconds'], rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('observation cost differs from the immutable measured ledger')
        environments.add(row.get('environment_hash'))
        by_run[run_id] = row
        ordered.append((stamp.timestamp(), position, row))
    if not ordered or len(environments) != 1 or None in environments:
        raise ValueError('nonempty measured table from one declared environment required')
    chosen = {}; provenance = []
    for _, position, row in sorted(ordered, key=lambda item: item[:2]):
        key = observation_key(row)
        if key not in chosen:
            chosen[key] = copy.deepcopy(row)
            provenance.append({'config_scenario_key': key, 'selected_run_id': row['run_id'],
                               'started_at_utc': starts[row['run_id']][1]['started_at_utc'],
                               'ledger_begin_position': position,
                               'replayed_source_accelerator_seconds': row['cost']['accelerator_seconds']})
    table = [chosen[key] for key in sorted(chosen)]
    missing = sorted(set(expected) - set(chosen))
    coverage = {'status': 'full_frozen_space' if not missing else 'restricted_to_observed_configurations',
                'expected_configurations': len(expected), 'observed_configurations': len(chosen),
                'missing_config_scenario_keys': missing,
                'original_live_rows': len(by_run), 'first_observation_provenance': provenance,
                'selection_rule': 'earliest completed live search reservation start per configuration, with ledger position breaking timestamp ties; no selection by outcome',
                'environment_hash': next(iter(environments)),
                'linked_trace_config_groups': len(connected_observation_groups(table))}
    return table, coverage


def run_ablation(table, coverage, *, seeds=(11, 29, 47), neural_epochs=200):
    """Six paired starts + up to four choices; policies see revealed rows only."""
    if any(r.get('source') != 'measured' for r in table):
        raise ValueError('ablation accepts measured evidence only')
    if any(r.get('partition') in {'holdout', 'test'} or r.get('workload', {}).get('partition') in {'holdout', 'test'} for r in table):
        raise ValueError('held-out outcomes are forbidden')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('distinct nonempty replay seeds required')
    by_key = {observation_key(row): row for row in table}
    if len(by_key) != len(table):
        raise ValueError('one static outcome per configuration is required')
    allowed = {'config_id', 'scenario_id', 'config', 'workload', 'is_default', 'known_valid'}
    candidates = [{k: copy.deepcopy(v) for k, v in by_key[key].items() if k in allowed}
                  for key in sorted(by_key)]
    result = {'label': LABEL, 'mode': MODE, 'evidence_source': 'measured',
              'is_live_optimizer_experiment': False, 'new_hardware_measurements': 0,
              'new_accelerator_seconds_charged': 0.0, 'coverage': coverage,
              'table_sha256': digest(table), 'seeds': list(seeds), 'neural_epochs': neural_epochs,
              'changed_parameter': 'neural optimistic acquisition beta: 1.0 versus 0.0',
              'initial_points': 6, 'requested_adaptive_points': 4, 'arms': [], 'paired_seed_results': [],
              'cost_interpretation': 'Sum of existing source reservation costs along each simulated path. These replay debits are never appended to the global real budget ledger.',
              'limitations': [
                  'Post-hoc empirical-table sensitivity analysis: it cannot establish a new hardware speedup or replace the live optimizer comparison.',
                  'One first-chronological observation freezes configuration-specific measurement noise. The table was adaptively sampled by the original searches.',
                  'Missing configurations are excluded, never imputed or selected. Restricted coverage changes the replay search space.',
                  'Original shared trace/config identities remain connected in training splits; fixed question content is reused across the profiling workload.',
                  'Three replay seeds quantify simulated choice variation conditional on one table, not independent hardware repetitions.',
                  'The six-start plus four-adaptive comparison is trial-matched. Recorded replay costs can differ; no equal actual tuning-cost claim is made.']}
    if len(candidates) <= 6:
        result.update(status='insufficient_observed_configurations_for_six_starts_and_adaptation')
        return result
    for seed in seeds:
        common = initial_design(candidates, 6, seed)
        initial_rows = [by_key[observation_key(c)] for c in common]
        try:
            grouped_split(initial_rows, seed)
        except ValueError as exc:
            result['paired_seed_results'].append({'seed': seed, 'status': 'skipped_invalid_grouped_split', 'reason': str(exc)})
            continue
        pair = []
        for beta in (1.0, 0.0):
            policy = SearchPolicy('neural', seed=seed, beta=beta, neural_epochs=neural_epochs,
                                  minimum_neural_groups=6, neural_target_standardization=True)
            observed = []; curve = []; replay_cost = 0.0; optimizer_cpu = 0.0
            for index in range(min(10, len(candidates))):
                if index < 6:
                    candidate = common[index]
                    decision = {'method_used': 'paired_common_initial_design', 'optimizer_cpu_seconds': 0.0,
                                'observations_seen': len(observed)}
                else:
                    candidate = policy.choose(candidates, observed)
                    if candidate is None:
                        break
                    decision = copy.deepcopy(policy.last_decision)
                # Sole outcome lookup: happens after the policy has committed.
                row = copy.deepcopy(by_key[observation_key(candidate)])
                visible_run_ids = [r['run_id'] for r in observed]
                observed.append(row)
                replay_cost += row['cost']['accelerator_seconds']
                optimizer_cpu += decision.get('optimizer_cpu_seconds', 0.0)
                best = measured_incumbent(observed)
                point = {'mode': MODE, 'seed': seed, 'beta': beta, 'trial': index + 1,
                         'visible_run_ids_before_proposal': visible_run_ids,
                         'revealed_source_run_id': row['run_id'], 'selected_config_id': row['config_id'],
                         'replayed_source_accelerator_seconds': row['cost']['accelerator_seconds'],
                         'cumulative_replay_debit_seconds': replay_cost, 'decision': decision,
                         'best_observed': best}
                if index >= 6 and policy.surrogate is not None:
                    point['actual_training_report'] = copy.deepcopy(policy.surrogate.report)
                curve.append(point)
            arm = {'mode': MODE, 'seed': seed, 'beta': beta,
                   'policy_settings': copy.deepcopy(policy.settings),
                   'initial_config_ids': [c['config_id'] for c in common],
                   'initial_source_run_ids': [r['run_id'] for r in initial_rows],
                   'trials': len(observed), 'adaptive_trials': max(0, len(observed) - 6),
                   'replay_debit_seconds': replay_cost, 'actual_cpu_optimizer_seconds': optimizer_cpu,
                   'best_observed': measured_incumbent(observed), 'curve': curve}
            result['arms'].append(arm); pair.append(arm)
        def score(arm):
            return arm['best_observed']['quality_goodput'] if arm['best_observed'] else None
        one, zero = map(score, pair)
        result['paired_seed_results'].append({'seed': seed, 'status': 'completed',
            'beta1_best_observed_goodput': one, 'beta0_best_observed_goodput': zero,
            'beta1_minus_beta0_goodput': one - zero if one is not None and zero is not None else None,
            'same_initial_configurations': pair[0]['initial_config_ids'] == pair[1]['initial_config_ids'],
            'same_initial_source_outcomes': pair[0]['initial_source_run_ids'] == pair[1]['initial_source_run_ids']})
    result['status'] = ('no_replay_seed_had_valid_grouped_split' if not result['arms'] else
                        'completed_posthoc_replay' if len(candidates) >= 10 else
                        'restricted_posthoc_replay_with_fewer_than_four_adaptive_choices')
    return result


def main():
    parser = argparse.ArgumentParser(description=LABEL)
    parser.add_argument('--observations', type=Path, nargs='+', required=True,
                        help='Only measured per-method live SEARCH observation journals; exclude final evaluation journals.')
    parser.add_argument('--budget-ledger', type=Path, required=True, help='Read-only reservation chronology/cost provenance.')
    parser.add_argument('--study', type=Path, required=True, help='Frozen cloud study with the full 12-configuration space.')
    parser.add_argument('--output', type=Path, required=True, help='Fresh JSON path for the explicitly post-hoc result.')
    parser.add_argument('--neural-epochs', type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve the existing ablation; choose a fresh output path')
    import torch
    torch.set_default_device('cpu'); torch.set_num_threads(1)
    from vllm_tuner.experiment import candidates_for
    study = json.loads(args.study.read_text())
    if len(study['workloads']) != 1:
        raise ValueError('one declared live workload required')
    expected = candidates_for(study, study['workloads'][0])
    if len(expected) != 12:
        raise ValueError('this post-hoc complement expects the full 12-configuration cloud grid')
    observations = [row for path in args.observations for row in read_jsonl(path)]
    table, coverage = first_chronological_table(observations, read_jsonl(args.budget_ledger), expected)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    result = run_ablation(table, coverage, neural_epochs=args.neural_epochs)
    result.update(actual_cpu_seconds=time.process_time() - started_cpu,
                  actual_cpu_wall_seconds=time.perf_counter() - started_wall,
                  frozen_study_sha256=digest(study),
                  observation_inputs=[str(path.resolve()) for path in args.observations],
                  read_only_budget_ledger=str(args.budget_ledger.resolve()))
    write_json(args.output, result, immutable=True)
    print(json.dumps({'label': LABEL, 'status': result['status'], 'coverage': coverage['status'],
                      'observed_configurations': coverage['observed_configurations'],
                      'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
