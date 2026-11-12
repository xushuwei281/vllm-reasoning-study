"""No real model/accelerator: audit the final experiment's control contracts."""
import copy
import json
from pathlib import Path

import pytest

from vllm_tuner import fixed_compute as fc
from vllm_tuner.data import DATA_REVISION, digest, freeze_protocol
from vllm_tuner.statistics import (EXACT_NONINFERIORITY_METHOD, paired_accuracy_ci,
                                   paired_accuracy_noninferiority)


class Tokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert kwargs['enable_thinking'] is True
        return messages[0]['content'] + '\n<think>'

    def encode(self, text, **kwargs):
        return list(range(10))


def protocol():
    shared = {'source': 'measured', 'partition': 'development_validation',
              'quality_gate_pass': True, 'question_count': 150,
              'question_ids_hash': digest(['dev'+str(i) for i in range(150)]),
              'sampling_seeds_hash': digest([1, 2]),
              'metrics': {'feasible': True, 'accuracy': .8, 'error_rate': 0.0}}
    selections = []
    for arm in ('default', 'tuned'):
        independent = copy.deepcopy(shared)
        independent['run_ids'] = [f'independent-{arm}']
        if arm == 'tuned':
            pairs = [{'question_id': f'dev{i}', 'sampling_seed': seed, 'correct': i < 120}
                     for i in range(150) for seed in (1, 2)]
            independent['paired_accuracy_ci'] = paired_accuracy_ci(pairs, pairs, bootstrap_samples=100)
            independent['paired_noninferiority'] = paired_accuracy_noninferiority(pairs, pairs)
        selections.append({'run_id': f'selection-{arm}', 'config_id': arm,
                           'scenario_id': 'dev', 'config': {'tensor_parallel_size': 1},
                           'workload': {'arrival_rate': .1}, 'source': 'measured',
                           'partition': 'train', 'environment_hash': 'environment-a',
                           'is_default': arm == 'default', 'metrics': {'feasible': True},
                           'independent_validation': independent})
    return {'study': {'final': {'reasoning_allowances': [16, 32], 'deadline_s': 10,
                               'question_count': 3, 'sampling_seeds': [724, 725]},
                      'model': {'max_model_len': 100}, 'budget': {'allocated_devices': 2},
                      'constraints': {'max_accuracy_drop': .02, 'max_error_rate': .01},
                      'quality_gate': {'method': EXACT_NONINFERIORITY_METHOD, 'question_count': 150,
                                       'confidence': .95, 'margin': .02},
                      'decoding': {'seed': 724, 'temperature': .6}},
            'selected': selections, 'data_revision': DATA_REVISION,
            'test_question_ids': [f'gsm8k:test:{i:05d}' for i in range(3)],
            'parser_version': 'strict-final-v1', 'test_results_seen': False}


def write_freeze(tmp_path, frozen=None):
    frozen = copy.deepcopy(frozen or protocol())
    frozen['freeze_hash'] = digest(frozen)
    path = tmp_path / 'freeze.json'
    path.write_text(json.dumps(frozen))
    return path


def official_rows():
    return [{'question_id': f'gsm8k:test:{i:05d}', 'question': f'Compute task {i}.',
             'gold': f'#### 98765{i}'} for i in range(3)]


def invocation(tmp_path, freeze, tokenizer):
    return dict(freeze_path=freeze, data_dir=tmp_path / 'data', tokenizer=tokenizer,
                capabilities={}, environment={'environment_hash': 'environment-a'},
                executable='unused-vllm', output_root=tmp_path / 'out', seed=725)


def test_four_arms_share_questions_actual_sampling_seed_and_full_denominator(tmp_path, monkeypatch):
    freeze = write_freeze(tmp_path)
    tokenizer = Tokenizer()
    calls = []
    monkeypatch.setattr(fc, 'select_questions', lambda directory, partition: official_rows())

    def run_trial(**kwargs):
        calls.append(copy.deepcopy({k: v for k, v in kwargs.items() if k != 'tokenizer'}))
        allowance = kwargs['candidate']['workload']['output_token_limit']
        tuned = not kwargs['candidate']['is_default']
        run_id = kwargs['run_id']
        path = Path(kwargs['output_root']) / 'runs' / run_id
        path.mkdir(parents=True)
        # Question 2 is never dispatched; it must remain in every denominator.
        rows = [{'question_id': kwargs['questions'][0]['question_id'], 'sampling_seed': 725,
                 'status': 'ok', 'finish_reason': 'stop', 'correct': True,
                 'completion_s': 2, 'output_tokens': 4},
                {'question_id': kwargs['questions'][1]['question_id'], 'sampling_seed': 726,
                 'status': 'timeout', 'finish_reason': None, 'correct': False,
                 'completion_s': 10, 'output_tokens': None}]
        (path / 'scored.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        wall = 11 + int(tuned) + allowance / 16
        return {'run_id': run_id, 'status': 'ok',
                'cost': {'wall_seconds': wall, 'accelerator_seconds': 2 * wall}}

    monkeypatch.setattr(fc, 'run_trial', run_trial)
    arms = fc.run_fixed_compute(**invocation(tmp_path, freeze, tokenizer))
    assert len(arms) == len(calls) == 4
    assert {(a['arm'], a['allowance']) for a in arms} == {
        ('default', 16), ('default', 32), ('tuned', 16), ('tuned', 32)}
    expected_ids = [row['question_id'] for row in official_rows()]
    for call, arm in zip(calls, arms):
        assert [q['question_id'] for q in call['questions']] == expected_ids
        assert call['seed'] == call['study']['decoding']['seed'] == 725
        assert call['study']['budget']['allocated_devices'] == 2
        assert call['reference_accuracy'] == .8
        assert call['candidate']['workload']['partition'] == 'holdout'
        assert call['phase'] == 'validation' and call['deadline_mode'] is True
        assert call['protocol_override']['measurement_s'] == call['protocol_override']['timeout_s'] == 10
        assert call['trace'] == [{'phase': 'measure', 'intended_arrival_s': 0.0}] * 3
        assert all('98765' not in prompt for prompt in call['prompts'])
        summary = arm['summary']
        assert summary['cohort_size'] == 3 and summary['unoffered_questions'] == 1
        assert summary['accuracy'] == pytest.approx(1 / 3)
        assert summary['deadline_constrained_accuracy'] == pytest.approx(1 / 3)
        assert summary['generated_tokens'] is None
        assert [r['sampling_seed'] for r in summary['question_outcomes']] == [725, 726, 727]
        assert 'whole' in arm['cost_match_note'].lower()
    assert len({a['actual_accelerator_seconds'] for a in arms}) > 1
    assert (tmp_path / 'out' / 'fixed-compute-summary-seed725.json').exists()
    plan = json.loads((tmp_path / 'out' / 'fixed-compute-plan-seed725.json').read_text())
    assert plan['sampling_seeds_by_question'] == [725, 726, 727]
    assert 'gold' not in json.dumps(plan)
    comparisons = json.loads((tmp_path / 'out' / 'fixed-compute-comparisons-seed725.json').read_text())
    assert len(comparisons) == 2
    assert all(not r['accuracy_difference']['quality_gate_usable'] for r in comparisons)
    assert all('reservations can differ' in r['interpretation'] for r in comparisons)


@pytest.mark.parametrize('mutation', [
    lambda f: f['selected'][1]['independent_validation'].update(quality_gate_pass=False),
    lambda f: f['selected'][1].update(source='synthetic'),
    lambda f: f['selected'][1].update(partition='holdout'),
    lambda f: f['selected'][1]['metrics'].update(feasible=False),
    lambda f: f['selected'][1]['independent_validation']['paired_noninferiority'].update(ci=[-.01, 1.0]),
    lambda f: f['selected'][1]['independent_validation']['paired_noninferiority'].update(quality_gate_usable=False),
    lambda f: f['selected'][1]['independent_validation']['paired_noninferiority'].update(harmful_questions=1),
    lambda f: f['selected'][1]['independent_validation']['paired_noninferiority'].update(planned_question_count=256),
    lambda f: f['selected'][1]['independent_validation']['paired_noninferiority'].update(reference_accuracy=.9),
    lambda f: f['selected'][1]['independent_validation']['paired_noninferiority'].update(confidence=.9),
    lambda f: f['study']['quality_gate'].update(question_count=256),
    lambda f: f['study'].pop('quality_gate'),
    lambda f: f['selected'][1]['independent_validation'].update(question_ids_hash=digest(['different'])),
    lambda f: f['selected'][1]['independent_validation'].update(run_ids=['selection-tuned']),
    lambda f: f['selected'][0]['independent_validation']['metrics'].update(accuracy=0),
    lambda f: f.update(data_revision='wrong'),
    lambda f: f.update(test_results_seen=True),
    lambda f: f['test_question_ids'].__setitem__(1, f['test_question_ids'][0]),
    lambda f: f['study']['final'].update(question_count=4),
])
def test_quality_and_preregistration_fail_before_test_access(tmp_path, monkeypatch, mutation):
    frozen = protocol()
    mutation(frozen)
    path = write_freeze(tmp_path, frozen)
    monkeypatch.setattr(fc, 'select_questions', lambda *args: pytest.fail('test answers read before gate'))
    monkeypatch.setattr(fc, 'run_trial', lambda **kwargs: pytest.fail('run launched before gate'))
    with pytest.raises(ValueError):
        fc.run_fixed_compute(**invocation(tmp_path, path, Tokenizer()))


def test_exact_gate_accepts_identical_pairs_without_hiding_degenerate_bootstrap():
    frozen = protocol()
    bootstrap = frozen['selected'][1]['independent_validation']['paired_accuracy_ci']
    assert bootstrap['degenerate_bootstrap'] and not bootstrap['quality_gate_usable']
    assert bootstrap['ci'] == [0, 0]
    fc.validate_frozen_protocol(frozen, require_integrity=False)


def empirical_protocol_with_offsetting_gain_and_harm():
    frozen = protocol()
    frozen['study']['quality_gate']['execution_criterion'] = 'empirical-paired-accuracy'
    reference = [{'question_id': f'dev{i}', 'sampling_seed': 724 + i, 'correct': i < 120}
                 for i in range(150)]
    candidate = [dict(row) for row in reference]
    candidate[0]['correct'] = False
    candidate[149]['correct'] = True
    evidence = frozen['selected'][1]['independent_validation']
    evidence['paired_accuracy_ci'] = paired_accuracy_ci(reference, candidate, bootstrap_samples=100)
    evidence['paired_noninferiority'] = paired_accuracy_noninferiority(reference, candidate)
    evidence['quality_claim_status'] = 'empirical_pass_statistical_noninferiority_inconclusive'
    return frozen


def test_declared_empirical_gate_allows_measurement_without_claiming_noninferiority():
    frozen = empirical_protocol_with_offsetting_gain_and_harm()
    evidence = frozen['selected'][1]['independent_validation']
    assert evidence['paired_noninferiority']['harmful_questions'] == 1
    assert evidence['paired_noninferiority']['quality_gate_pass'] is False
    assert evidence['paired_noninferiority']['difference_candidate_minus_reference'] == 0
    fc.validate_frozen_protocol(frozen, require_integrity=False)
    # Missing the explicit preregistered branch restores the stricter gate.
    frozen['study']['quality_gate'].pop('execution_criterion')
    with pytest.raises(ValueError, match='bound does not pass'):
        fc.validate_frozen_protocol(frozen, require_integrity=False)


def test_empirical_gate_cannot_hide_uncertainty_or_relax_two_point_margin():
    frozen = empirical_protocol_with_offsetting_gain_and_harm()
    frozen['selected'][1]['independent_validation']['quality_claim_status'] = 'statistical_noninferiority_established'
    with pytest.raises(ValueError, match='claim status'):
        fc.validate_frozen_protocol(frozen, require_integrity=False)
    frozen = empirical_protocol_with_offsetting_gain_and_harm()
    frozen['selected'][1]['independent_validation']['metrics']['accuracy'] = .77
    for field in ('paired_accuracy_ci', 'paired_noninferiority'):
        frozen['selected'][1]['independent_validation'][field]['candidate_accuracy'] = .77
    frozen['selected'][1]['independent_validation']['paired_noninferiority']['difference_candidate_minus_reference'] = -.03
    with pytest.raises(ValueError):
        fc.validate_frozen_protocol(frozen, require_integrity=False)


def test_optional_test_cohort_seed_is_reproducible_without_test_content(tmp_path):
    config = protocol()
    legacy = freeze_protocol(tmp_path / 'legacy.json', config['study'], config['selected'], {})
    assert legacy['test_question_ids'][:3] == [f'gsm8k:test:{i:05d}' for i in range(3)]
    config['study']['final']['test_cohort_seed'] = 724
    first = freeze_protocol(tmp_path / 'first.json', config['study'], config['selected'], {})
    repeated = freeze_protocol(tmp_path / 'repeated.json', config['study'], config['selected'], {})
    assert first['test_question_ids'] == repeated['test_question_ids']
    assert set(first['test_question_ids']) == set(legacy['test_question_ids'])
    assert first['test_question_ids'][:3] != legacy['test_question_ids'][:3]
    changed = copy.deepcopy(first)
    changed['test_question_ids'][0], changed['test_question_ids'][1] = changed['test_question_ids'][1], changed['test_question_ids'][0]
    with pytest.raises(ValueError, match='cohort seed'):
        fc.validate_frozen_protocol(changed, require_integrity=False)
    config['study']['final']['test_cohort_seed'] = 725
    other = freeze_protocol(tmp_path / 'other.json', config['study'], config['selected'], {})
    assert first['test_question_ids'][:3] != other['test_question_ids'][:3]


def test_tampered_freeze_and_unfrozen_seed_fail_before_test_access(tmp_path, monkeypatch):
    path = write_freeze(tmp_path)
    monkeypatch.setattr(fc, 'select_questions', lambda *args: pytest.fail('test answers read'))
    inputs = invocation(tmp_path, path, Tokenizer())
    inputs['seed'] = 999
    with pytest.raises(ValueError, match='seed'):
        fc.run_fixed_compute(**inputs)
    changed = json.loads(path.read_text())
    changed['study']['final']['deadline_s'] = 11
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='modified'):
        fc.run_fixed_compute(**invocation(tmp_path, path, Tokenizer()))


def test_failed_arms_keep_full_cohort_and_zero_actual_cost(tmp_path, monkeypatch):
    path = write_freeze(tmp_path)
    monkeypatch.setattr(fc, 'select_questions', lambda *args: official_rows())
    monkeypatch.setattr(fc, 'run_trial', lambda **kw: {
        'run_id': kw['run_id'], 'status': 'known_invalid',
        'cost': {'wall_seconds': 0.0, 'accelerator_seconds': 0.0}})
    arms = fc.run_fixed_compute(**invocation(tmp_path, path, Tokenizer()))
    assert len(arms) == 4
    for arm in arms:
        assert arm['execution_status'] == 'known_invalid'
        assert arm['summary']['cohort_size'] == arm['summary']['unoffered_questions'] == 3
        assert arm['summary']['accuracy'] == 0
        assert arm['actual_accelerator_seconds'] == 0
        assert arm['summary']['correct_per_accelerator_second'] is None


def test_overlength_frozen_test_prompts_abort_without_adapting_or_running(tmp_path, monkeypatch):
    frozen = protocol()
    frozen['study']['model']['max_model_len'] = 40
    path = write_freeze(tmp_path, frozen)
    monkeypatch.setattr(fc, 'select_questions', lambda *args: official_rows())
    monkeypatch.setattr(fc, 'run_trial', lambda **kw: pytest.fail('overlength arm launched'))
    with pytest.raises(ValueError, match='context'):
        fc.run_fixed_compute(**invocation(tmp_path, path, Tokenizer()))


def test_runtime_environment_must_match_development_before_test_access(tmp_path, monkeypatch):
    path = write_freeze(tmp_path)
    monkeypatch.setattr(fc, 'select_questions', lambda *args: pytest.fail('test answers read'))
    inputs = invocation(tmp_path, path, Tokenizer())
    inputs['environment'] = {'environment_hash': 'unseen-hardware'}
    with pytest.raises(ValueError, match='environment'):
        fc.run_fixed_compute(**inputs)


def test_failed_collector_retains_offered_and_partial_request_evidence(tmp_path, monkeypatch):
    freeze = write_freeze(tmp_path)
    monkeypatch.setattr(fc, 'select_questions', lambda *args: official_rows())

    def failed(**kwargs):
        path = Path(kwargs['output_root']) / 'runs' / kwargs['run_id']
        path.mkdir(parents=True)
        offered = [{'question_id': f'gsm8k:test:{i:05d}', 'request_id': f'r{i}'} for i in range(2)]
        (path / 'offered.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in offered))
        partial = {**offered[0], 'sampling_seed': 725, 'status': 'timeout',
                   'finish_reason': None, 'text': 'unfinished thought', 'output_tokens': None,
                   'completion_s': 10}
        (path / 'requests.jsonl').write_text(json.dumps(partial) + '\n')
        return {'run_id': kwargs['run_id'], 'status': 'failed',
                'cost': {'wall_seconds': 11, 'accelerator_seconds': 22}}

    monkeypatch.setattr(fc, 'run_trial', failed)
    arms = fc.run_fixed_compute(**invocation(tmp_path, freeze, Tokenizer()))
    assert all(a['summary']['offered_questions'] == 2 for a in arms)
    assert all(a['summary']['unoffered_questions'] == 1 for a in arms)
    assert all(a['summary']['generated_tokens'] is None for a in arms)
    assert all(a['summary']['accuracy'] == 0 for a in arms)
