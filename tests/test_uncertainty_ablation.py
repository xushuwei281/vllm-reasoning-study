"""CPU fixture checks for the explicitly post-hoc script; no research measurements."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'ablate_uncertainty.py'
spec = importlib.util.spec_from_file_location('ablate_uncertainty', SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def fixture(count=12):
    rows = []; events = []
    stamp = datetime(2026, 11, 3, tzinfo=timezone.utc)
    for i in range(count):
        rows.append({'run_id': f'fixture-run-{i}', 'config_id': f'c{i:02d}', 'scenario_id': 's',
                     'source': 'measured', 'partition': 'train', 'status': 'ok', 'environment_hash': 'TEST_FIXTURE',
                     'trace_id': f't{i}', 'trace_hash': f't{i}',
                     'config': {'precision': 'bf16', 'max_num_batched_tokens': 512 * (1 + i % 3),
                                'max_num_seqs': 8 * (1 + i // 3), 'gpu_memory_utilization': .8, 'tensor_parallel_size': 1},
                     'workload': {'profile': 'short', 'arrival_rate': .6, 'input_tokens_mean': 100,
                                  'input_tokens_p95': 130, 'output_token_limit': 4096, 'burstiness': .08},
                     'metrics': {'quality_goodput': .1 + i / 100, 'p99_e2e_s': 30 + i, 'feasible': i % 4 != 0},
                     'cost': {'accelerator_seconds': 300 + i / 1000, 'wall_seconds': 300 + i / 1000}})
        events.extend([{'event': 'begin', 'run_id': rows[-1]['run_id'], 'phase': 'search', 'devices': 1,
                        'started_at_utc': (stamp + timedelta(minutes=i * 5)).isoformat()},
                       {'event': 'finish', 'run_id': rows[-1]['run_id'], 'accelerator_seconds': 300 + i / 1000}])
    expected = [{k: copy.deepcopy(v) for k, v in r.items() if k in {'config_id', 'scenario_id', 'config', 'workload'}} for r in rows]
    return rows, events, expected


def test_first_chronological_selection_is_not_file_order_or_best_outcome():
    rows, events, expected = fixture()
    later = copy.deepcopy(rows[0]); later['run_id'] = 'later-and-better'; later['metrics']['quality_goodput'] = 999
    events.extend([{'event': 'begin', 'run_id': later['run_id'], 'phase': 'search', 'devices': 1,
                    'started_at_utc': '2026-11-04T00:00:00+00:00'},
                   {'event': 'finish', 'run_id': later['run_id'], 'accelerator_seconds': 300}])
    table, coverage = ablation.first_chronological_table([later] + list(reversed(rows)), events, expected)
    assert table[0]['run_id'] == rows[0]['run_id']
    assert table[0]['metrics']['quality_goodput'] == .1
    assert coverage['status'] == 'full_frozen_space'
    assert coverage['original_live_rows'] == 13
    restricted, info = ablation.first_chronological_table(rows[:8], events, expected)
    assert len(restricted) == 8 and info['status'] == 'restricted_to_observed_configurations'
    assert len(info['missing_config_scenario_keys']) == 4


def test_source_holdout_and_ledger_provenance_separation():
    rows, events, expected = fixture()
    for field, value, message in [('source', 'synthetic', 'measured evidence'),
                                   ('partition', 'holdout', 'held-out')]:
        altered = copy.deepcopy(rows); altered[0][field] = value
        with pytest.raises(ValueError, match=message):
            ablation.first_chronological_table(altered, events, expected)
    altered = copy.deepcopy(rows); altered[0]['cost']['accelerator_seconds'] = 1
    with pytest.raises(ValueError, match='differs from'):
        ablation.first_chronological_table(altered, events, expected)
    with pytest.raises(ValueError, match='completed live reservation'):
        ablation.first_chronological_table(rows, events[2:], expected)


def test_paired_real_training_has_no_hidden_outcomes_or_unobserved_cost_gate(monkeypatch):
    torch.set_num_threads(1)
    rows, events, expected = fixture()
    table, coverage = ablation.first_chronological_table(rows, events, expected)
    original_choose = ablation.SearchPolicy.choose
    visible_calls = []
    def audited_choose(self, candidates, observed):
        assert all(set(c) <= {'config_id', 'scenario_id', 'config', 'workload', 'is_default', 'known_valid'} for c in candidates)
        assert all('metrics' not in c and 'cost' not in c and 'run_id' not in c for c in candidates)
        visible_calls.append((self.settings['seed'], self.settings['beta'], len(observed)))
        return original_choose(self, candidates, observed)
    monkeypatch.setattr(ablation.SearchPolicy, 'choose', audited_choose)
    before = copy.deepcopy(table)
    result = ablation.run_ablation(table, coverage, neural_epochs=4)
    assert table == before
    assert result['mode'] == ablation.MODE and result['is_live_optimizer_experiment'] is False
    assert result['new_accelerator_seconds_charged'] == 0
    assert len(result['arms']) == 6 and len(visible_calls) == 24
    assert {count for _, _, count in visible_calls} == {6, 7, 8, 9}
    for paired in result['paired_seed_results']:
        assert paired['same_initial_configurations'] and paired['same_initial_source_outcomes']
    for arm in result['arms']:
        assert arm['adaptive_trials'] == 4
        assert arm['replay_debit_seconds'] == pytest.approx(sum(p['replayed_source_accelerator_seconds'] for p in arm['curve']))
        for point in arm['curve'][6:]:
            assert point['decision']['method_used'] == 'neural'
            assert point['revealed_source_run_id'] not in point['visible_run_ids_before_proposal']
            assert point['actual_training_report']['members'][0]['epochs']
    # A hidden label/cost can only affect decisions after that row is revealed.
    initial_ids = set(result['arms'][0]['initial_config_ids'])
    altered = copy.deepcopy(table)
    for row in altered:
        if row['config_id'] not in initial_ids:
            row['metrics']['quality_goodput'] = 999
            row['cost']['accelerator_seconds'] = 99999
    changed = ablation.run_ablation(altered, coverage, seeds=(11,), neural_epochs=4)
    for first, second in zip(result['arms'][:2], changed['arms']):
        assert first['curve'][6]['selected_config_id'] == second['curve'][6]['selected_config_id']
        assert second['trials'] == 10  # No peek at a hidden source cost to admit a choice.


def test_trace_links_are_preserved_and_invalid_initial_grouping_is_skipped():
    rows, events, expected = fixture()
    for row in rows:
        row['trace_id'] = row['trace_hash'] = 'one-shared-trace'
    table, coverage = ablation.first_chronological_table(rows, events, expected)
    result = ablation.run_ablation(table, coverage, neural_epochs=4)
    assert coverage['linked_trace_config_groups'] == 1
    assert not result['arms']
    assert all(p['status'] == 'skipped_invalid_grouped_split' for p in result['paired_seed_results'])


def test_partial_coverage_does_not_invent_four_adaptive_observations():
    torch.set_num_threads(1)
    rows, events, expected = fixture()
    table, coverage = ablation.first_chronological_table(rows[:7], events, expected)
    result = ablation.run_ablation(table, coverage, seeds=(11,), neural_epochs=2)
    assert result['coverage']['status'] == 'restricted_to_observed_configurations'
    assert result['status'] == 'restricted_posthoc_replay_with_fewer_than_four_adaptive_choices'
    assert all(arm['trials'] == 7 and arm['adaptive_trials'] == 1 for arm in result['arms'])
    selected = {p['selected_config_id'] for arm in result['arms'] for p in arm['curve']}
    assert selected <= {r['config_id'] for r in rows[:7]}
