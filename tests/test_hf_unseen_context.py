"""CPU-only optional diagnostic tests; no test dataset or accelerator is used."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('hf_unseen_context_test_module', SCRIPTS / 'hf_unseen_context.py')
uc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uc)


def ingredients():
    study = json.loads((SCRIPTS.parent / 'configs/study-cloud.json').read_text())
    search = json.loads((SCRIPTS.parent / 'results/continuation/search-plan.json').read_text())
    rows = [{'question_id': f'gsm8k:train:{i:05d}', 'question': f'Question {i}', 'gold': '#### 1'}
            for i in range(256)]
    manifest = {'partitions': {'calibration': ['gsm8k:train:06000'], 'profiling': ['gsm8k:train:06001'],
                              'development_validation': [r['question_id'] for r in rows]}}
    return study, search, rows, manifest


def plan_for(study, search, rows, manifest):
    return uc.make_plan(study, search, rows, manifest, registered_at='2026-11-03T12:00:00+00:00',
                        source_hash=hashlib.sha256((SCRIPTS / 'hf_unseen_context.py').read_bytes()).hexdigest())


def test_preregistration_is_disjoint_and_copies_trace_without_outcomes():
    study, search, rows, manifest = ingredients()
    plan = plan_for(study, search, rows, manifest)
    assert plan['question_ids'] == [r['question_id'] for r in rows[150:206]]
    assert not set(plan['question_ids']) & {r['question_id'] for r in rows[:150]}
    assert len(plan['measurement_question_ids']) == 54
    assert plan['trace'] == search['traces'][0] and plan['trace'] is not search['traces'][0]
    assert plan['sampling_seeds_by_question'] == list(range(13724, 13780))
    assert plan['decoding'] == {**study['decoding'], 'seed': 13724}
    assert plan['status'] == 'preregistered_not_run' and 'arms' not in plan
    manifest['partitions']['profiling'].append(rows[150]['question_id'])
    with pytest.raises(ValueError, match='unused independent'):
        plan_for(study, search, rows, manifest)


def allocation_and_ledger():
    allocation = {'created_at': datetime.fromtimestamp(0, timezone.utc).isoformat(), 'devices': 1,
                  'timeout_seconds': 14400, 'preallocation_charged_accelerator_seconds': 100}
    ledger = {'spent_accelerator_seconds': 12100, 'remaining_accelerator_seconds': 16700,
              'active_run_ids': []}
    return allocation, ledger


def test_time_gate_protects_export_tail_and_unreconciled_global_cost():
    allocation, ledger = allocation_and_ledger()
    allowed = uc.resource_gate(allocation, ledger, now=13210, core_complete=True)
    assert allowed['allowed'] and allowed['job_seconds_remaining'] == 1190
    assert allowed['unreconciled_allocation_seconds'] == 1210
    assert allowed['effective_budget_seconds_remaining'] == 15490
    late = uc.resource_gate(allocation, ledger, now=13210.001, core_complete=True)
    assert not late['allowed'] and 'export' in late['reasons'][0]
    ledger['remaining_accelerator_seconds'] = 2399
    insufficient = uc.resource_gate(allocation, ledger, now=13210, core_complete=True)
    assert not insufficient['allowed'] and 'original-ledger' in insufficient['reasons'][0]
    assert not uc.resource_gate(allocation, ledger, now=12000, core_complete=False)['allowed']


def test_gate_blocks_missing_core_or_active_queue():
    allocation, ledger = allocation_and_ledger()
    ledger['active_run_ids'] = ['main-live-run']
    gate = uc.resource_gate(allocation, ledger, now=12000, core_complete=True)
    assert not gate['allowed'] and gate['reasons'] == ['another reservation is active']


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is True
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return list(range(600))


@pytest.fixture
def optional_context(tmp_path, monkeypatch):
    study, search, rows, manifest = ingredients()
    output, data = tmp_path / 'results', tmp_path / 'data'
    output.mkdir()
    data.mkdir()
    monkeypatch.setattr(uc.continuation, 'OUT', output)
    monkeypatch.setattr(uc.continuation, 'DATA', data)
    monkeypatch.setattr(uc.continuation, 'LEDGER', tmp_path / 'original-budget.jsonl')
    plan = plan_for(study, search, rows, manifest)
    (output / 'unseen-context-plan.json').write_text(json.dumps(plan))
    (data / 'train-manifest.json').write_text(json.dumps(manifest))
    selected = [{'is_default': arm == 'default', 'run_id': f'selection-{arm}',
                 'source': 'measured', 'partition': 'train', 'config_id': arm,
                 'config': {'max_num_seqs': 256 if arm == 'default' else 16},
                 'workload': {'profile': 'short', 'arrival_rate': .6, 'output_token_limit': 4096},
                 'environment_hash': 'environment-a'} for arm in ['default', 'tuned']]
    frozen = {'study': study, 'selected': selected, 'freeze_hash': 'immutable-core-freeze'}
    freeze_bytes = json.dumps(frozen)
    (output / 'freeze.json').write_text(freeze_bytes)
    monkeypatch.setattr(uc, 'validate_frozen_protocol', lambda value: value)
    monkeypatch.setattr(uc.continuation, 'context', lambda: (copy.deepcopy(study),
                        {'environment_hash': 'environment-a'}, {}, Tokenizer()))

    def select(directory, partition):
        assert partition == 'development_validation', 'optional diagnostic attempted test access'
        return copy.deepcopy(rows)

    monkeypatch.setattr(uc, 'select_questions', select)
    monkeypatch.setattr(uc, '_admission', lambda count: {'allowed': True, 'arms_remaining': count})
    return output, plan, frozen, freeze_bytes


def test_run_preserves_selected_configs_and_uses_long_prompts_with_paired_new_seed(optional_context, monkeypatch):
    output, plan, frozen, freeze_bytes = optional_context
    calls = []

    def trial(**kwargs):
        calls.append(kwargs)
        assert kwargs['phase'] == 'validation' and kwargs['ledger_path'] == uc.continuation.LEDGER
        assert kwargs['study']['decoding']['seed'] == kwargs['seed'] == 13724
        assert all(p.startswith('Context for a school newsletter:') for p in kwargs['prompts'])
        assert kwargs['candidate']['workload']['profile'] == 'long_context_stress'
        assert kwargs['candidate']['workload']['input_tokens_max'] == 600
        assert kwargs['candidate']['config'] == frozen['selected'][len(calls)-1]['config']
        directory = output / 'runs' / kwargs['run_id']
        directory.mkdir(parents=True)
        rows = [{'question_id': q['question_id'], 'phase': item['phase'], 'sampling_seed': 13724+i,
                 'status': 'ok', 'finish_reason': 'stop', 'correct': True, 'completion_s': 99}
                for i, (q, item) in enumerate(zip(kwargs['questions'], kwargs['trace']))]
        (directory / 'scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return {'run_id': kwargs['run_id'], 'source': 'measured', 'status': 'ok',
                'metrics': {'accuracy': 1., 'quality_goodput': .6, 'peak_memory_bytes': 1},
                'cost': {'accelerator_seconds': 300., 'wall_seconds': 300.}}

    monkeypatch.setattr(uc, 'run_trial', trial)
    result = uc.run()
    assert result['status'] == 'completed' and len(calls) == 2
    assert result['actual_accelerator_seconds'] == 600
    assert result['paired_accuracy_difference']['question_clusters'] == 54
    assert calls[0]['reference_accuracy'] is None and calls[1]['reference_accuracy'] == 1
    assert (output / 'freeze.json').read_text() == freeze_bytes
    assert json.loads((output / 'unseen-context-plan.json').read_text()) == plan
    assert uc.run() == result and len(calls) == 2


def test_insufficient_time_records_not_run_without_context_or_trial(optional_context, monkeypatch):
    output, plan, frozen, freeze_bytes = optional_context
    monkeypatch.setattr(uc, '_admission', lambda count: {'allowed': False, 'reasons': ['export tail protected']})
    monkeypatch.setattr(uc.continuation, 'context', lambda: pytest.fail('context loaded despite time gate'))
    monkeypatch.setattr(uc, 'run_trial', lambda **kw: pytest.fail('GPU trial launched despite time gate'))
    result = uc.run()
    assert result['status'] == 'not_run' and result['arms'] == []
    assert result['reason'] == 'export tail protected'
    assert (output / 'freeze.json').read_text() == freeze_bytes


def test_missing_measurement_response_stays_in_denominator(optional_context):
    output, plan, frozen, freeze_bytes = optional_context
    outcomes = uc.measurement_outcomes([], plan)
    assert len(outcomes) == 54 and sum(row['correct'] for row in outcomes) == 0
    assert all(not row['recorded'] for row in outcomes)
    assert outcomes[0]['sampling_seed'] == 13726


def test_modified_preregistration_is_rejected_before_gpu_admission(optional_context, monkeypatch):
    output, plan, frozen, freeze_bytes = optional_context
    plan['protocol']['timeout_s'] = 300
    (output / 'unseen-context-plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(uc, '_admission', lambda count: pytest.fail('tampered plan reached admission'))
    with pytest.raises(ValueError, match='modified'):
        uc.run()
