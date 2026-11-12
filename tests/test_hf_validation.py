"""CPU-only orchestration checks; synthetic fixtures never enter study results."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('hf_validation_test_module', SCRIPTS / 'hf_validation.py')
hv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hv)


def observation(config_id, qag=.2, run_id=None):
    return {'source': 'measured', 'status': 'ok', 'partition': 'train',
            'environment_hash': 'environment-a', 'run_id': run_id or f'selection-{config_id}',
            'config_id': config_id, 'config': {'max_num_seqs': 64},
            'workload': {'scenario_id': 'development', 'output_token_limit': 4096, 'arrival_rate': .4},
            'metrics': {'feasible': True, 'quality_goodput': qag, 'accuracy': .8, 'error_rate': 0}}


def test_finalist_selection_has_deterministic_ties_and_no_infeasible_rows():
    histories = {'random': [observation('b')], 'gp': [observation('a')],
                 'neural': [observation('c', qag=.9)]}
    histories['neural'][0]['metrics']['feasible'] = False
    method, selected = hv.select_finalist(histories, 'environment-a')
    assert method == 'gp' and selected['config_id'] == 'a'
    histories['neural'][0]['source'] = 'synthetic'
    with pytest.raises(ValueError, match='measured observations'):
        hv.select_finalist(histories, 'environment-a')


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is True
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return [1] * 20


@pytest.fixture
def staged(tmp_path, monkeypatch):
    output, data = tmp_path / 'results', tmp_path / 'data'
    output.mkdir()
    data.mkdir()
    monkeypatch.setattr(hv.continuation, 'OUT', output)
    monkeypatch.setattr(hv.continuation, 'DATA', data)
    monkeypatch.setattr(hv.continuation, 'LEDGER', tmp_path / 'original-ledger.jsonl')
    study = json.loads((SCRIPTS.parent / 'configs' / 'study-cloud.json').read_text())
    study['quality_gate']['execution_criterion'] = 'empirical-paired-accuracy'
    monkeypatch.setattr(hv.continuation, 'context', lambda: (copy.deepcopy(study),
                        {'environment_hash': 'environment-a'}, {}, Tokenizer()))
    questions = [{'question_id': f'gsm8k:train:{i:05d}', 'question': f'Problem {i}', 'gold': '#### 1'}
                 for i in range(150)]

    def select_questions(directory, partition):
        assert partition == 'development_validation', 'test data accessed during validation'
        return questions

    monkeypatch.setattr(hv, 'select_questions', select_questions)
    monkeypatch.setattr(hv, 'prepare_dataset', lambda *a, **kw: pytest.fail('test data fetched during validation'))
    (data / 'train-manifest.json').write_text(json.dumps({'partitions': {
        'profiling': ['gsm8k:train:05000'], 'calibration': ['gsm8k:train:05001'],
        'development_validation': [q['question_id'] for q in questions]}}))
    for method, config in [('random', 'b'), ('gp', 'a'), ('neural', 'c')]:
        directory = output / 'search' / method
        directory.mkdir(parents=True)
        (directory / 'observations.jsonl').write_text(json.dumps(observation(config)) + '\n')
    (output / 'default-reference.json').write_text(json.dumps(observation('default')))
    calls = []

    def install_runner(candidate_correct=120, offsetting=True):
        def trial(study, environment, caps, tokenizer, config, workload, selected, run_id, **kwargs):
            assert (output / 'independent-validation-plan.json').exists()
            plan = json.loads((output / 'independent-validation-plan.json').read_text())
            assert len(plan['question_ids']) == len(selected) == 150
            assert plan['sampling_seeds_by_question'] == list(range(724, 874))
            assert plan['protocol'] == hv.VALIDATION_PROTOCOL
            assert kwargs['trace'][-1]['intended_arrival_s'] == pytest.approx(149 / .6)
            assert kwargs['phase'] == 'validation' and kwargs['seed'] == 724
            assert study['decoding'] == plan['decoding']
            assert workload['output_token_limit'] == 4096
            assert workload['arrival_rate'] == .6
            calls.append((run_id, copy.deepcopy(kwargs)))
            is_default = kwargs['default']
            count = 120 if is_default else candidate_correct
            correct = [i < count for i in range(150)]
            if not is_default and offsetting:
                correct[0], correct[149] = False, True
            rows = [{'question_id': q['question_id'], 'phase': 'measure', 'sampling_seed': 724 + i,
                     'status': 'ok', 'finish_reason': 'stop', 'correct': correct[i]}
                    for i, q in enumerate(selected)]
            directory = output / 'runs' / run_id
            directory.mkdir(parents=True)
            (directory / 'scored.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            row = observation('validation', run_id=run_id)
            row['metrics']['accuracy'] = sum(correct) / 150
            row['metrics']['feasible'] = row['metrics']['accuracy'] >= .8 - .02
            return row
        monkeypatch.setattr(hv.continuation, 'trial', trial)

    return output, calls, install_runner


def test_validation_freezes_before_runs_and_retains_inconclusive_statistics(staged):
    output, calls, install_runner = staged
    install_runner()
    status = hv.validate()
    assert len(calls) == 2
    assert calls[0][1]['reference'] is None and calls[1][1]['reference'] == .8
    assert status['final_allowed'] and not status['official_test_accessed']
    assert status['quality_claim_status'] == 'empirical_pass_statistical_noninferiority_inconclusive'
    frozen = json.loads((output / 'freeze.json').read_text())
    candidate = frozen['selected'][1]
    assert candidate['run_id'] == 'selection-a'
    assert candidate['independent_validation']['paired_noninferiority']['quality_gate_pass'] is False
    assert candidate['independent_validation']['quality_gate_pass'] is True
    assert (output / 'independent-default-outcomes.json').exists()


def test_validation_failure_has_explicit_status_and_cannot_create_test_freeze(staged):
    output, calls, install_runner = staged
    install_runner(candidate_correct=116, offsetting=False)
    status = hv.validate()
    assert len(calls) == 2
    assert status['status'] == 'quality_gate_failed' and not status['final_allowed']
    assert not (output / 'freeze.json').exists()
    assert json.loads((output / 'validation-status.json').read_text()) == status


def test_missing_validation_question_cannot_disappear_from_quality_denominator():
    rows = [{'question_id': 'a', 'phase': 'measure', 'sampling_seed': 724,
             'status': 'ok', 'finish_reason': 'stop', 'correct': True}]
    with pytest.raises(ValueError, match='complete frozen question cohort'):
        hv.paired_outcomes(rows, ['a', 'b'], [724, 725])


def test_final_stage_requires_matching_freeze_before_dataset_fetch(staged, monkeypatch):
    output, calls, install_runner = staged
    install_runner()
    hv.validate()
    status_path = output / 'validation-status.json'
    status = json.loads(status_path.read_text())
    status['freeze_hash'] = 'wrong'
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match='matching its freeze'):
        hv.final()


def test_final_stage_uses_frozen_seed_and_original_ledger(staged, monkeypatch):
    from vllm_tuner import fixed_compute
    output, calls, install_runner = staged
    install_runner()
    hv.validate()
    events = []

    def prepare(directory, split, *, freeze_path):
        assert split == 'test' and freeze_path == output / 'freeze.json'
        events.append('test-preparation')

    def fixed(freeze_path, **kwargs):
        assert events == ['test-preparation']
        assert freeze_path == output / 'freeze.json'
        assert kwargs['ledger_path'] == hv.continuation.LEDGER
        assert kwargs['seed'] == 724 and kwargs['output_root'] == output / 'final'
        events.append('four-arm-runner')
        return [{'synthetic_test_fixture': True}] * 4

    monkeypatch.setattr(hv, 'prepare_dataset', prepare)
    monkeypatch.setattr(fixed_compute, 'run_fixed_compute', fixed)
    assert len(hv.final()) == 4
    assert events == ['test-preparation', 'four-arm-runner']
