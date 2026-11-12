"""Fairness tests use a recording policy so they cannot conceal oracle access."""
import copy
import json
from pathlib import Path
import random

import pytest

import vllm_tuner.replay as module


class RecordingPolicy:
    calls = []

    def __init__(self, method, seed, **kwargs):
        self.method, self.seed = method, seed
        self.last_decision = {}

    def choose(self, candidates, observed):
        self.calls.append({'method': self.method, 'seed': self.seed,
                           'observed': copy.deepcopy(observed), 'candidates': copy.deepcopy(candidates)})
        assert all('metrics' not in c and 'cost' not in c and 'run_id' not in c for c in candidates)
        used = {r['config_id'] for r in observed}
        remaining = [c for c in candidates if c['config_id'] not in used]
        if self.method == 'default':
            remaining = [c for c in remaining if c.get('is_default')]
        self.last_decision = {'method_used': self.method, 'optimizer_cpu_seconds': .25}
        return remaining[0] if remaining else None

    def save(self, path):
        Path(path).write_text(json.dumps({'method': self.method}))


@pytest.fixture
def recording_policy(monkeypatch):
    RecordingPolicy.calls = []
    monkeypatch.setattr(module, 'SearchPolicy', RecordingPolicy)
    return RecordingPolicy


def test_fixture_is_not_measured_and_repeats_remain_together():
    table = module.synthetic_table()
    assert len(table) == 144
    assert {r['source'] for r in table} == {'synthetic'}
    assert {r['repetition'] for r in table} == {0, 1}
    partitions = {}
    for row in table:
        key = row['config_id'], row['scenario_id']
        partitions.setdefault(key, set()).add(row['partition'])
    assert all(len(values) == 1 for values in partitions.values())
    assert {r['scenario_id'] for r in table if r['partition'] == 'holdout'} == {'unseen-mix'}


def test_equal_common_starts_cost_and_no_unpaid_oracle_access(tmp_path, recording_policy):
    table = module.synthetic_table()
    curves, summaries = module.replay(table, tmp_path / 'replay', budget_s=450,
                                      maximum_trial_cost_s=100, initial_points=2, seeds=(11, 29))
    for seed in (11, 29):
        methods = [[r for r in curves if r['seed'] == seed and r['method'] == method]
                   for method in ('random', 'gp', 'neural')]
        assert all(len(rows) == 4 for rows in methods)
        assert len({tuple(r['selected_config_id'] for r in rows[:2]) for rows in methods}) == 1
        assert all(rows[-1]['cumulative_accelerator_seconds'] == 360 for rows in methods)
        for rows in methods:
            assert len({r['selected_config_id'] for r in rows}) == len(rows)
            assert all(r['selected_run_id'].endswith('-0') for r in rows)
    for call in recording_policy.calls:
        prior = [r for r in curves if r['seed'] == call['seed'] and r['method'] == call['method']]
        expected = [r['selected_config_id'] for r in prior[:len(call['observed'])]]
        assert [r['config_id'] for r in call['observed']] == expected
        assert all(r['scenario_id'] == 'mid' and r['partition'] == 'train' for r in call['observed'])
    assert all(s['cost_s'] <= 450 for s in summaries)
    assert all(s['trials'] == 1 for s in summaries if s['method'] == 'default')
    assert all(s['cost_interpretation'] == 'prescribed synthetic fixture' for s in summaries)


def test_replay_is_invariant_to_raw_row_order(tmp_path, recording_policy):
    table = module.synthetic_table()
    shuffled = copy.deepcopy(table)
    random.Random(82).shuffle(shuffled)
    a, _ = module.replay(table, tmp_path / 'a', seeds=(11,), initial_points=3, budget_s=450)
    b, _ = module.replay(shuffled, tmp_path / 'b', seeds=(11,), initial_points=3, budget_s=450)
    assert a == b


def test_failed_outcomes_consume_budget(tmp_path, recording_policy):
    curves, _ = module.replay(module.synthetic_table(), tmp_path / 'failure', scenario='high',
                              seeds=(11,), initial_points=1, budget_s=100)
    assert len(curves) == 4
    assert all(c['cumulative_accelerator_seconds'] == 90 and c['best_feasible_goodput'] == 0 for c in curves)


def test_invalid_budget_mixed_sources_duplicate_and_missing_repetition(tmp_path):
    table = module.synthetic_table()
    with pytest.raises(ValueError, match='positive maximum'):
        module.replay(table, tmp_path / 'zero', maximum_trial_cost_s=0)
    mixed = copy.deepcopy(table)
    mixed[-1]['source'] = 'measured'  # even cross-partition mixing is prohibited
    with pytest.raises(ValueError, match='mix measured'):
        module.replay(mixed, tmp_path / 'mixed')
    with pytest.raises(ValueError, match='duplicate'):
        module.replay(table + [copy.deepcopy(next(r for r in table if r['scenario_id'] == 'mid' and r['repetition'] == 0))], tmp_path / 'dup')
    with pytest.raises(ValueError, match='no development candidates'):
        module.replay(table, tmp_path / 'rep', repetition=50)
    with pytest.raises(ValueError, match='bound'):
        module.replay(table, tmp_path / 'cost', maximum_trial_cost_s=89)
    with pytest.raises(ValueError, match='no development candidates'):
        module.replay(table, tmp_path / 'holdout', scenario='unseen-mix')


def test_does_not_overwrite_completed_replay(tmp_path, recording_policy):
    path = tmp_path / 'existing'
    module.replay(module.synthetic_table(), path, seeds=(1,), budget_s=100)
    original = (path / 'summary.json').read_bytes()
    with pytest.raises(FileExistsError, match='nonempty'):
        module.replay(module.synthetic_table(), path, seeds=(1,), budget_s=100)
    assert original == (path / 'summary.json').read_bytes()


def test_diagnostic_requires_real_holdout_and_same_evidence_source(tmp_path):
    table = module.synthetic_table()
    with pytest.raises(ValueError, match='both development'):
        module.train_diagnostic([r for r in table if r['partition'] == 'train'], tmp_path / 'nohold')
    table[-1]['source'] = 'measured'
    with pytest.raises(ValueError, match='mix measured'):
        module.train_diagnostic(table, tmp_path / 'mixed')
