"""Live-loop contracts tested with fake measured-shaped rows; no hardware runs."""
import copy
import json

import pytest

import vllm_tuner.experiment as experiment
from vllm_tuner.accounting import RunJournal
from vllm_tuner.search import SearchPolicy


def candidates():
    return [
        {'config_id': f'c{i}', 'scenario_id': 's', 'is_default': i == 0, 'known_valid': True,
         'config': {'precision': 'bf16', 'max_num_batched_tokens': 256 * (i + 1),
                    'max_num_seqs': 1, 'gpu_memory_utilization': .5, 'tensor_parallel_size': 1},
         'workload': {'profile': 'short', 'arrival_rate': .1, 'input_tokens_mean': 100,
                      'input_tokens_p95': 150, 'output_token_limit': 1024, 'burstiness': 1}}
        for i in range(3)
    ]


def outcome(candidate, index, cost=100):
    # Tests construct schema fixtures; these are never saved as research results.
    return {**copy.deepcopy(candidate), 'run_id': f'run-{index}', 'repetition': 0,
            'source': 'measured', 'partition': 'train', 'environment_hash': 'TEST_FIXTURE',
            'status': 'ok', 'metrics': {'quality_goodput': .1 + index / 100, 'p99_e2e_s': 1., 'feasible': True},
            'cost': {'accelerator_seconds': cost, 'wall_seconds': cost, 'optimizer_cpu_seconds': 0.}}


def test_admission_allows_exact_maximum_and_stops_without_free_extra_trial(tmp_path):
    calls = []
    def runner(candidate, index):
        calls.append(index)
        return outcome(candidate, index)
    rows = experiment.closed_loop(candidates(), runner, tmp_path / 'equal', method='random',
                                  initial_points=1, accelerator_budget_s=100, maximum_trial_cost_s=100)
    assert calls == [0] and len(rows) == 1
    assert rows[0]['cost']['accelerator_seconds'] == 100
    experiment.closed_loop(candidates(), runner, tmp_path / 'too-small', method='random',
                           initial_points=1, accelerator_budget_s=99, maximum_trial_cost_s=100)
    assert calls == [0]


def test_default_does_not_get_exploratory_common_starts(tmp_path):
    calls = []
    def runner(candidate, index):
        calls.append(candidate['config_id'])
        return outcome(candidate, index)
    rows = experiment.closed_loop(candidates(), runner, tmp_path, method='default',
                                  initial_points=6, accelerator_budget_s=1000, maximum_trial_cost_s=100)
    assert calls == ['c0'] and len(rows) == 1
    assert rows[0]['optimizer_decision']['method_used'] == 'default'


def test_cpu_cost_recorded_and_only_paid_observations_visible(tmp_path, monkeypatch):
    original = SearchPolicy.choose
    visible = []
    def recording(self, choices, observed):
        assert all('metrics' not in c and 'cost' not in c for c in choices)
        visible.append([r['run_id'] for r in observed])
        selected = original(self, choices, observed)
        self.last_decision.update(optimizer_cpu_seconds=1.25, optimizer_wall_seconds=1.5)
        return selected
    monkeypatch.setattr(SearchPolicy, 'choose', recording)
    rows = experiment.closed_loop(candidates(), outcome, tmp_path, method='random', initial_points=0,
                                  accelerator_budget_s=200, maximum_trial_cost_s=100)
    assert visible == [[], ['run-0']]
    assert all(r['cost']['optimizer_cpu_seconds'] == 1.25 for r in rows)
    assert all(r['cost']['optimizer_wall_seconds'] == 1.5 for r in rows)
    assert sum(r['cost']['accelerator_seconds'] for r in rows) == 200  # CPU was unreserved
    summary = json.loads((tmp_path / 'incumbent.json').read_text())
    assert summary['optimizer_cpu_seconds'] == 2.5


def test_resume_does_not_repeat_idempotent_runner(tmp_path):
    calls = []
    def runner(candidate, index):
        calls.append(index)
        return outcome(candidate, index)
    kwargs = dict(method='random', initial_points=1, accelerator_budget_s=200, maximum_trial_cost_s=100)
    first = experiment.closed_loop(candidates(), runner, tmp_path, **kwargs)
    second = experiment.closed_loop(candidates(), runner, tmp_path, **kwargs)
    assert first == second and calls == [0, 1]
    with pytest.raises(FileExistsError, match='immutable'):
        experiment.closed_loop(candidates(), runner, tmp_path, **{**kwargs, 'seed': 99})


def test_pending_requires_explicit_completed_record_reconciliation(tmp_path):
    live = RunJournal(tmp_path / 'live' / 'observations.jsonl')
    output = tmp_path / 'search'
    calls = []
    def crashed(candidate, index):
        calls.append(index)
        live.append(outcome(candidate, index))
        raise RuntimeError('injected crash after live result, before search append')
    kwargs = dict(method='random', initial_points=1, accelerator_budget_s=200, maximum_trial_cost_s=100)
    with pytest.raises(RuntimeError, match='injected crash'):
        experiment.closed_loop(candidates(), crashed, output, **kwargs)
    with pytest.raises(RuntimeError, match='pending proposal'):
        experiment.closed_loop(candidates(), crashed, output, **kwargs)
    assert calls == [0]
    with pytest.raises(ValueError, match='exactly one completed'):
        experiment.recover_pending(output, live.path, 'absent')
    recovered = experiment.recover_pending(output, live.path, 'run-0')
    assert experiment.recover_pending(output, live.path, 'run-0') == recovered
    assert len(RunJournal(output / 'observations.jsonl').records()) == 1
    def resumed(candidate, index):
        calls.append(index)
        row = outcome(candidate, index)
        live.append(row)
        return row
    rows = experiment.closed_loop(candidates(), resumed, output, **kwargs)
    assert calls == [0, 1] and len(rows) == 2
    assert sum(r['cost']['accelerator_seconds'] for r in rows) == 200


def test_recovery_after_append_is_idempotent(tmp_path, monkeypatch):
    live = RunJournal(tmp_path / 'live.jsonl'); output = tmp_path / 'search'
    original = experiment._save_incumbent
    def crash(*args):
        raise RuntimeError('after append')
    monkeypatch.setattr(experiment, '_save_incumbent', crash)
    def runner(candidate, index):
        row = outcome(candidate, index); live.append(row); return row
    with pytest.raises(RuntimeError, match='after append'):
        experiment.closed_loop(candidates(), runner, output, method='default', accelerator_budget_s=100, maximum_trial_cost_s=100)
    monkeypatch.setattr(experiment, '_save_incumbent', original)
    experiment.recover_pending(output, live.path, 'run-0')
    assert len(RunJournal(output / 'observations.jsonl').records()) == 1
    assert not (output / 'pending.json').exists()


@pytest.mark.parametrize('field,value', [('source', 'synthetic'), ('partition', 'holdout'), ('partition', 'test')])
def test_runner_source_and_partition_checked_before_search_visibility(tmp_path, field, value):
    def bad(candidate, index):
        row = outcome(candidate, index); row[field] = value; return row
    with pytest.raises(ValueError, match='measured development'):
        experiment.closed_loop(candidates(), bad, tmp_path, method='random', initial_points=1,
                               accelerator_budget_s=100, maximum_trial_cost_s=100)
    assert RunJournal(tmp_path / 'observations.jsonl').records() == []
    assert (tmp_path / 'pending.json').exists()


def test_candidate_cannot_contain_unobserved_answers_or_holdout(tmp_path):
    choices = candidates(); choices[0]['metrics'] = {'quality_goodput': 9999}
    with pytest.raises(ValueError, match='unobserved outcomes'):
        experiment.closed_loop(choices, outcome, tmp_path / 'oracle', method='random')
    choices = candidates(); choices[0]['workload']['partition'] = 'holdout'
    with pytest.raises(ValueError, match='held-out'):
        experiment.closed_loop(choices, outcome, tmp_path / 'holdout', method='random')


def test_duplicate_runner_run_id_and_cost_overrun_are_not_free(tmp_path):
    def duplicated(candidate, index):
        row = outcome(candidate, index); row['run_id'] = 'always-same'; return row
    with pytest.raises(ValueError, match='already charged'):
        experiment.closed_loop(candidates(), duplicated, tmp_path / 'duplicate', method='random', initial_points=0,
                               accelerator_budget_s=200, maximum_trial_cost_s=100)
    output = tmp_path / 'overrun'
    kwargs = dict(method='default', accelerator_budget_s=100, maximum_trial_cost_s=100)
    with pytest.raises(RuntimeError, match='exceeded declared bound'):
        experiment.closed_loop(candidates(), lambda c, i: outcome(c, i, cost=120), output, **kwargs)
    assert RunJournal(output / 'observations.jsonl').records()[0]['cost']['accelerator_seconds'] == 120
    with pytest.raises(RuntimeError, match='previous trial exceeded'):
        experiment.closed_loop(candidates(), outcome, output, **kwargs)


def test_reservation_bound_counts_every_reserved_device_and_shutdown_wait():
    study = {'startup_timeout_s': 10, 'benchmark': {'warmup_s': 1, 'ramp_s': 2, 'measurement_s': 3, 'drain_s': 4},
             'budget': {'allocated_devices': 2}}
    bound = experiment.trial_reservation_bound(study)
    assert bound == {'wall_seconds': 65., 'accelerator_seconds': 130.}
    assert experiment.trial_reservation_bound(study, {'measurement_s': 13})['accelerator_seconds'] == 150


def test_live_loop_attaches_actual_neural_proposal_model_before_runner(tmp_path):
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    choices = []
    for i in range(8):
        choice = copy.deepcopy(candidates()[0])
        choice.update(config_id=f'c{i}', is_default=i == 0)
        choice['config']['max_num_batched_tokens'] = 256 * (i + 1)
        choices.append(choice)
    def runner(candidate, index):
        if index == 6:
            pending = json.loads((tmp_path / 'pending.json').read_text())
            artifact = pending['decision']['model_artifact']
            assert Path(artifact['checkpoint']).is_file()
            snapshot = json.loads((Path(artifact['directory']) / 'observations.json').read_text())
            assert len(snapshot) == 6 and all(r['run_id'] != 'run-6' for r in snapshot)
        return outcome(candidate, index)
    rows = experiment.closed_loop(choices, runner, tmp_path, method='neural', initial_points=6,
                                  neural_epochs=4, accelerator_budget_s=700, maximum_trial_cost_s=100)
    assert len(rows) == 7
    assert rows[-1]['optimizer_decision']['method_used'] == 'neural'
    assert rows[-1]['optimizer_decision']['model_artifact']['checkpoint'].endswith('ensemble.pt')
