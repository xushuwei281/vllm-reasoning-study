"""Optional preregistered context generalization diagnostic after the core test.

This script never reads official test data, selects a new configuration or
provisions resources. The coordinator invokes ``run`` only after the four-arm
core experiment; admission depends on completion/integrity and resources alone.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import hf_continuation as continuation
from vllm_tuner.accounting import BudgetLedger
from vllm_tuner.benchmark import evaluate_collected, validate_trace
from vllm_tuner.data import digest, prompt_for, read_jsonl, select_questions, write_json
from vllm_tuner.experiment import run_trial, trial_reservation_bound
from vllm_tuner.fixed_compute import validate_frozen_protocol
from vllm_tuner.statistics import paired_accuracy_ci

BASE_SEED = 13724
TRIAL_BOUND_SECONDS = 445
EXPORT_RESERVE_SECONDS = 300
JOB_LIMIT_SECONDS = 14400
PROTOCOL = {"warmup_s": 5, "ramp_s": 5, "measurement_s": 90,
            "drain_s": 120, "timeout_s": 120, "reserved_wall_seconds": 300,
            "max_in_flight": 512, "warmup_output_token_limit": 64}


def read_json(path):
    return json.loads(Path(path).read_text())


def make_plan(study, search_plan, development_rows, manifest, *, registered_at, source_hash):
    """Pure plan construction: only training metadata and predeclared settings."""
    if (study['model']['max_model_len'] != 8192 or study['decoding']['max_tokens'] != 4096
            or study['budget']['allocated_devices'] != 1):
        raise ValueError('optional context diagnostic retains 8192 context, 4096 allowance and one device')
    if search_plan['study_hash'] != digest(study):
        raise ValueError('study differs from the frozen main search plan')
    rows = development_rows[150:206]
    ids = [row['question_id'] for row in rows]
    excluded = {row['question_id'] for row in development_rows[:150]}
    excluded |= set(manifest['partitions']['calibration']) | set(manifest['partitions']['profiling'])
    if len(ids) != 56 or len(set(ids)) != 56 or set(ids) & excluded:
        raise ValueError('optional cohort must contain 56 unused independent development questions')
    if any(not qid.startswith('gsm8k:train:') for qid in ids):
        raise ValueError('optional diagnostic uses official training questions only')
    trace = copy.deepcopy(search_plan['traces'][0])
    validate_trace(trace, PROTOCOL)
    if (len(trace) != 56 or [r['phase'] for r in trace[:2]] != ['warmup', 'ramp']
            or any(r['phase'] != 'measure' for r in trace[2:])):
        raise ValueError('copied trace must have one warmup, one ramp and 54 measurement requests')
    if trial_reservation_bound(study, PROTOCOL)['accelerator_seconds'] != TRIAL_BOUND_SECONDS:
        raise ValueError('optional trial admission differs from preregistered 445 seconds')
    plan = {'status': 'preregistered_not_run', 'registered_at_utc': registered_at,
            'role': 'secondary independent unseen-context diagnostic; cannot alter core selections or claims',
            'partition': 'development_validation', 'partition_slice': [150, 206],
            'question_ids': ids, 'measurement_question_ids': ids[2:],
            'question_ids_hash': digest(ids), 'measurement_question_ids_hash': digest(ids[2:]),
            'question_content_hashes': [digest(row['question']) for row in rows],
            'dataset_manifest_hash': digest(manifest), 'study_hash': digest(study),
            'main_search_plan_hash': digest(search_plan), 'script_sha256': source_hash,
            'profile': 'long_context_stress', 'prompt_implementation_sha256': hashlib.sha256(
                (continuation.ROOT / 'src/vllm_tuner/data.py').read_bytes()).hexdigest(),
            'output_token_limit': 4096, 'max_model_len': 8192,
            'decoding': {**study['decoding'], 'seed': BASE_SEED},
            'sampling_seeds_by_question': [BASE_SEED + i for i in range(56)],
            'protocol': copy.deepcopy(PROTOCOL), 'trace': trace, 'trace_hash': digest(trace),
            'arrival_rate': .6, 'arm_order': ['default', 'tuned'],
            'selection_rule': 'Use the two original core-freeze configurations unchanged; no reselection',
            'execution_gate': {'core_four_arm_summary_required': True,
                               'job_limit_from_created_seconds': JOB_LIMIT_SECONDS,
                               'per_arm_admission_seconds': TRIAL_BOUND_SECONDS,
                               'export_reserve_seconds': EXPORT_RESERVE_SECONDS,
                               'initial_required_remaining_seconds': 2 * TRIAL_BOUND_SECONDS + EXPORT_RESERVE_SECONDS,
                               'criterion': 'remaining time and original-ledger budget only; never core outcome values'},
            'analysis': 'Full 54-question measurement denominator; paired question bootstrap for accuracy and deadline accuracy; report QAG, latency/censoring, memory, errors and actual resource cost separately',
            'no_outcome_adaptation': True, 'official_test_data_required': False}
    plan['plan_hash'] = digest(plan)
    return plan


def validate_plan(plan):
    proof = copy.deepcopy(plan)
    claimed = proof.pop('plan_hash', None)
    if claimed is None or digest(proof) != claimed:
        raise ValueError('optional preregistration was modified or lacks an integrity hash')
    return plan


def preregister(data_dir=None):
    target = continuation.OUT / 'unseen-context-plan.json'
    if target.exists():
        return validate_plan(read_json(target))
    if (continuation.OUT / 'freeze.json').exists() or (continuation.OUT / 'final').exists():
        raise ValueError('new optional plan must precede the core final freeze and test outcomes')
    data_dir = Path(data_dir) if data_dir else continuation.DATA
    plan = make_plan(read_json(continuation.ROOT / 'configs/study-cloud.json'),
                     read_json(continuation.OUT / 'search-plan.json'),
                     select_questions(data_dir, 'development_validation'),
                     read_json(data_dir / 'train-manifest.json'),
                     registered_at=datetime.now(timezone.utc).isoformat(),
                     source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(target, plan, immutable=True)
    print('UNSEEN_CONTEXT_PREREGISTERED', json.dumps({'plan_hash': plan['plan_hash'], 'questions': 56}), flush=True)
    return plan


def resource_gate(allocation, ledger_summary, *, now, core_complete, arms_remaining=2):
    """Conservative read-only admission, including cloud time not yet reconciled."""
    created = datetime.fromisoformat(allocation['created_at']).timestamp()
    cap = min(JOB_LIMIT_SECONDS, allocation.get('timeout_seconds', JOB_LIMIT_SECONDS))
    seconds_left = created + cap - now
    devices = allocation['devices']
    elapsed = max(0.0, now - created) * devices
    cloud_charged = (ledger_summary['spent_accelerator_seconds']
                     - allocation['preallocation_charged_accelerator_seconds'])
    unreconciled = max(0.0, elapsed - cloud_charged)
    budget_left = ledger_summary['remaining_accelerator_seconds'] - unreconciled
    required = arms_remaining * TRIAL_BOUND_SECONDS + EXPORT_RESERVE_SECONDS
    reasons = []
    if not core_complete:
        reasons.append('core four-arm final summary is absent or incomplete')
    if devices != 1:
        reasons.append('allocation differs from the preregistered one device')
    if ledger_summary.get('active_run_ids'):
        reasons.append('another reservation is active')
    if seconds_left < required:
        reasons.append('insufficient job lifetime after reserving 300 seconds for export')
    if budget_left < required * devices:
        reasons.append('insufficient original-ledger budget including unreconciled allocated time and export')
    return {'allowed': not reasons, 'reasons': reasons, 'arms_remaining': arms_remaining,
            'job_seconds_remaining': seconds_left, 'effective_budget_seconds_remaining': budget_left,
            'unreconciled_allocation_seconds': unreconciled, 'required_seconds': required,
            'job_deadline_epoch': created + cap,
            'latest_optional_completion_epoch': created + cap - EXPORT_RESERVE_SECONDS}


def _core_complete():
    path = continuation.OUT / 'final/fixed-compute-summary-seed724.json'
    if not path.exists():
        return False
    rows = read_json(path)
    # Inspect completion/identity only, never outcomes to decide whether to run.
    return (len(rows) == 4 and {(r['arm'], r['allowance']) for r in rows}
            == {('default', 2048), ('default', 4096), ('tuned', 2048), ('tuned', 4096)})


def _admission(arms_remaining):
    if not continuation.LEDGER.exists():
        raise ValueError('the original global ledger is required; no replacement ledger may be created')
    ledger = BudgetLedger(continuation.LEDGER, 28800, .25)
    return resource_gate(read_json(continuation.OUT / 'cloud-allocation.json'), ledger.summary(),
                         now=time.time(), core_complete=_core_complete(), arms_remaining=arms_remaining)


def measurement_outcomes(rows, plan):
    expected = dict(zip(plan['question_ids'], plan['sampling_seeds_by_question']))
    phases = {qid: item['phase'] for qid, item in zip(plan['question_ids'], plan['trace'])}
    lookup = {}
    for row in rows:
        qid = row['question_id']
        if qid not in expected or qid in lookup or row.get('phase') != phases[qid]:
            raise ValueError('unexpected or duplicated optional diagnostic request')
        if row.get('sampling_seed') not in {None, expected[qid]}:
            raise ValueError('optional diagnostic response seed differs from the plan')
        if row.get('status') == 'ok' and row.get('sampling_seed') is None:
            raise ValueError('completed optional diagnostic response lacks seed provenance')
        lookup[qid] = row
    outcomes = []
    for qid in plan['measurement_question_ids']:
        row = lookup.get(qid, {})
        correct = bool(row.get('correct')) and row.get('status') == 'ok' and row.get('finish_reason') == 'stop'
        end = row.get('completion_s')
        timely = correct and isinstance(end, (int, float)) and math.isfinite(end) and end < 100
        outcomes.append({'question_id': qid, 'sampling_seed': expected[qid], 'correct': correct,
                         'deadline_correct': bool(timely), 'recorded': qid in lookup})
    return outcomes


def _save_summary(plan, status, arms, admissions, reason=None):
    summary = {'status': status, 'plan_hash': plan['plan_hash'], 'reason': reason,
               'secondary_only': True, 'core_selection_unchanged': True,
               'official_test_data_accessed': False, 'admission_checks': admissions, 'arms': arms,
               'actual_accelerator_seconds': sum(arm['cost']['accelerator_seconds'] for arm in arms),
               'interpretation': 'Independent unseen-context diagnostic; one run per arm, 54 measured questions, no robust P99 or population noninferiority claim'}
    if len(arms) == 2:
        reference, candidate = [arm['measurement_outcomes'] for arm in arms]
        summary['paired_accuracy_difference'] = paired_accuracy_ci(reference, candidate, seed=BASE_SEED)
        summary['paired_deadline_accuracy_difference'] = paired_accuracy_ci(
            [dict(r, correct=r['deadline_correct']) for r in reference],
            [dict(r, correct=r['deadline_correct']) for r in candidate], seed=BASE_SEED)
    write_json(continuation.OUT / 'unseen-context-summary.json', summary, immutable=True)
    print('UNSEEN_CONTEXT_FINISHED', json.dumps({'status': status, 'reason': reason, 'arms': len(arms)}), flush=True)
    return summary


def run():
    plan = validate_plan(read_json(continuation.OUT / 'unseen-context-plan.json'))
    summary_path = continuation.OUT / 'unseen-context-summary.json'
    if summary_path.exists():
        saved = read_json(summary_path)
        if saved['plan_hash'] != plan['plan_hash']:
            raise ValueError('optional summary belongs to a different preregistration')
        return saved
    first_admission = _admission(2)
    admissions = [first_admission]
    if not first_admission['allowed']:
        return _save_summary(plan, 'not_run', [], admissions, '; '.join(first_admission['reasons']))
    if plan['script_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('optional diagnostic implementation changed after preregistration')
    frozen = validate_frozen_protocol(read_json(continuation.OUT / 'freeze.json'))
    if digest(frozen['study']) != plan['study_hash']:
        raise ValueError('core study differs from the optional preregistration')
    study, environment, capabilities, tokenizer = continuation.context()
    if digest(study) != plan['study_hash']:
        raise ValueError('study changed after optional preregistration')
    if {r['environment_hash'] for r in frozen['selected']} != {environment['environment_hash']}:
        raise ValueError('optional diagnostic must use the validated deployment environment')
    if hashlib.sha256((continuation.ROOT / 'src/vllm_tuner/data.py').read_bytes()).hexdigest() != plan['prompt_implementation_sha256']:
        raise ValueError('prompt implementation changed after optional preregistration')
    manifest = read_json(continuation.DATA / 'train-manifest.json')
    questions = select_questions(continuation.DATA, 'development_validation')[150:206]
    if (digest(manifest) != plan['dataset_manifest_hash']
            or [q['question_id'] for q in questions] != plan['question_ids']
            or [digest(q['question']) for q in questions] != plan['question_content_hashes']):
        raise ValueError('optional independent cohort changed after preregistration')
    study = copy.deepcopy(study)
    study['decoding']['seed'] = BASE_SEED
    if study['decoding'] != plan['decoding']:
        raise ValueError('optional diagnostic may change only the declared sampling seed')
    prompts = [prompt_for(q['question'], tokenizer, profile='long_context_stress') for q in questions]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    if max(lengths) + 4096 > 8192:
        return _save_summary(plan, 'not_run', [], admissions, 'long-context prompts exceed frozen context; no truncation')
    selected = sorted(frozen['selected'], key=lambda row: not row['is_default'])
    execution = {'plan_hash': plan['plan_hash'], 'core_freeze_hash': frozen['freeze_hash'],
                 'selected_source_run_ids': [r['run_id'] for r in selected],
                 'selected_configs': [r['config'] for r in selected],
                 'prompt_hashes': [digest(p) for p in prompts], 'prompt_token_lengths': lengths,
                 'effective_study_hash': digest(study), 'environment_hash': environment['environment_hash']}
    write_json(continuation.OUT / 'unseen-context-execution-plan.json', execution, immutable=True)
    arms = []
    for index, selection in enumerate(selected):
        admission = _admission(2 - index)
        admissions.append(admission)
        if not admission['allowed']:
            return _save_summary(plan, 'partially_run' if arms else 'not_run', arms, admissions,
                                 '; '.join(admission['reasons']))
        arm = 'default' if selection['is_default'] else 'tuned'
        candidate = copy.deepcopy(selection)
        candidate['scenario_id'] = 'unseen-context-stress'
        candidate['workload'] = {**candidate['workload'], 'scenario_id': candidate['scenario_id'],
                                 'profile': 'long_context_stress', 'partition': 'development_validation',
                                 'arrival_rate': .6, 'output_token_limit': 4096,
                                 'input_tokens_max': max(lengths), 'input_tokens_mean': sum(lengths) / len(lengths),
                                 'input_tokens_p95': sorted(lengths)[math.ceil(.95 * len(lengths)) - 1]}
        row = run_trial(candidate=candidate, study=study, capabilities=capabilities, environment=environment,
                        questions=questions, prompts=prompts, output_root=continuation.OUT,
                        executable=shutil.which('vllm'), tokenizer=tokenizer, seed=BASE_SEED, phase='validation',
                        run_id=f'unseen-context-{arm}-seed{BASE_SEED}', port=8137,
                        ledger_path=continuation.LEDGER, protocol_override=plan['protocol'], trace=plan['trace'],
                        reference_accuracy=None if not arms else arms[0]['full_measurement_cohort_accuracy'])
        scored_path = continuation.OUT / 'runs' / row['run_id'] / 'scored.jsonl'
        if scored_path.exists():
            rows = read_jsonl(scored_path)
        else:
            raw_path = scored_path.with_name('requests.jsonl')
            rows = evaluate_collected(read_jsonl(raw_path), questions, tokenizer) if raw_path.exists() else []
        outcomes = measurement_outcomes(rows, plan)
        arms.append({'arm': arm, 'run_id': row['run_id'], 'source': row['source'],
                     'execution_status': row['status'], 'config': copy.deepcopy(selection['config']),
                     'metrics': row['metrics'], 'cost': row['cost'], 'measurement_cohort_size': 54,
                     'full_measurement_cohort_accuracy': sum(r['correct'] for r in outcomes) / 54,
                     'full_measurement_deadline_accuracy': sum(r['deadline_correct'] for r in outcomes) / 54,
                     'missing_measurement_records': sum(not r['recorded'] for r in outcomes),
                     'measurement_outcomes': outcomes})
    status = 'completed' if all(a['execution_status'] == 'ok' for a in arms) else 'completed_with_failed_runs'
    return _save_summary(plan, status, arms, admissions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preregister', 'run'])
    parser.add_argument('--data-dir', type=Path, help='training data location for preregistration only')
    args = parser.parse_args()
    if args.stage == 'preregister':
        preregister(args.data_dir)
    else:
        if args.data_dir:
            parser.error('--data-dir is only permitted for preregistration')
        run()


if __name__ == '__main__':
    main()
