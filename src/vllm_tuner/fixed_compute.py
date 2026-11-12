"""Four-arm fixed-cohort reasoning test, gated by independent development evidence.

The common service deadline is fixed; measured whole-reservation costs remain a
frontier unless those costs actually match. No final outcome changes later arms.
"""
from __future__ import annotations

import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping

from .data import DATA_REVISION, digest, select_questions, prompt_for, write_json, read_jsonl
from .experiment import run_trial
from .statistics import (EXACT_NONINFERIORITY_METHOD, fixed_compute_summary,
                         harmful_discordance_upper_bound, paired_accuracy_ci)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_frozen_protocol(frozen: Mapping[str, Any], *, require_integrity: bool = True) -> dict:
    """Validate final preregistration before reading or downloading test answers.

    This validates auditable evidence fields, not a cryptographic attestation of
    their scientific truth. Retain the referenced independent raw run records.
    A bare user-set quality_gate_pass boolean is deliberately insufficient.
    """
    proof = copy.deepcopy(dict(frozen))
    claimed = proof.pop('freeze_hash', None)
    if require_integrity and (claimed is None or digest(proof) != claimed):
        raise ValueError('frozen protocol modified or missing integrity hash')
    if frozen.get('data_revision') != DATA_REVISION:
        raise ValueError('frozen official dataset revision mismatch')
    if frozen.get('parser_version') != 'strict-final-v1' or frozen.get('test_results_seen') is not False:
        raise ValueError('freeze must declare the supported parser and no prior test results')
    selected = frozen.get('selected', [])
    if len(selected) != 2 or sum(bool(x.get('is_default')) for x in selected) != 1:
        raise ValueError('exactly one measured default and one measured finalist required')
    study = frozen['study']
    final = study['final']
    if not _positive_int(study['budget']['allocated_devices']):
        raise ValueError('final experiment requires a positive fixed accelerator allocation')
    allowances = final['reasoning_allowances']
    if len(allowances) != 2 or len(set(allowances)) != 2 or not all(_positive_int(v) for v in allowances):
        raise ValueError('two different positive development-selected allowances required')
    if not math.isfinite(final['deadline_s']) or final['deadline_s'] <= 0:
        raise ValueError('positive finite frozen service deadline required')
    count = final['question_count']
    ids = frozen.get('test_question_ids', [])
    if (not _positive_int(count) or count > len(ids) or len(set(ids)) != len(ids)
            or any(not isinstance(q, str) or not re.fullmatch(r'gsm8k:test:\d{5}', q)
                   or int(q.rsplit(':', 1)[1]) >= 1319 for q in ids)):
        raise ValueError('frozen test cohort must contain enough unique official question IDs')
    cohort_seed = final.get('test_cohort_seed')
    if cohort_seed is not None:
        if not isinstance(cohort_seed, int) or isinstance(cohort_seed, bool):
            raise ValueError('test cohort seed must be a preregistered integer')
        expected_ids = [f'gsm8k:test:{i:05d}' for i in range(1319)]
        random.Random(cohort_seed).shuffle(expected_ids)
        if ids[:count] != expected_ids[:count]:
            raise ValueError('frozen test IDs do not match the preregistered cohort seed')
    seeds = final.get('sampling_seeds', [study['decoding']['seed']])
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ValueError('frozen sampling seeds must be distinct integers')
    margin = study['constraints']['max_accuracy_drop']
    max_errors = study['constraints']['max_error_rate']
    if not 0 <= margin < 1 or not 0 <= max_errors < 1:
        raise ValueError('invalid frozen accuracy/error constraints')
    gate_plan = study.get('quality_gate', {})
    execution_criterion = gate_plan.get('execution_criterion', 'exact-noninferiority')
    if (gate_plan.get('method') != EXACT_NONINFERIORITY_METHOD
            or not _positive_int(gate_plan.get('question_count')) or gate_plan['question_count'] < 30
            or not _finite_number(gate_plan.get('confidence')) or not .95 <= gate_plan['confidence'] < 1
            or gate_plan.get('margin') != margin
            or execution_criterion not in {'exact-noninferiority', 'empirical-paired-accuracy'}):
        raise ValueError('a fixed exact paired quality-gate plan must be preregistered')
    evidence = []
    for row in selected:
        if (row.get('source') != 'measured' or row.get('partition') not in {'train', 'validation'}
                or row.get('workload', {}).get('partition') in {'holdout', 'test'}
                or not row.get('metrics', {}).get('feasible', False)):
            raise ValueError('final selections must be measured feasible development observations')
        validation = row.get('independent_validation', {})
        run_ids = validation.get('run_ids', [])
        metrics = validation.get('metrics', {})
        if (validation.get('quality_gate_pass') is not True or validation.get('source') != 'measured'
                or validation.get('partition') != 'development_validation' or not run_ids
                or len(set(run_ids)) != len(run_ids) or row.get('run_id') in run_ids
                or not _positive_int(validation.get('question_count'))
                or validation['question_count'] != gate_plan['question_count']
                or not metrics.get('feasible', False)
                or not isinstance(metrics.get('accuracy'), (float, int)) or not 0 < metrics['accuracy'] <= 1
                or not isinstance(metrics.get('error_rate'), (float, int)) or not 0 <= metrics['error_rate'] <= max_errors):
            raise ValueError('independent measured development quality evidence is required')
        for key in ('question_ids_hash', 'sampling_seeds_hash'):
            if not isinstance(validation.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', validation[key]):
                raise ValueError('independent quality evidence must identify the paired cohort and seeds')
        if not row.get('is_default'):
            gate = validation.get('paired_noninferiority', {})
            bounds = gate.get('ci', [])
            if (gate.get('method') != EXACT_NONINFERIORITY_METHOD
                    or gate.get('quality_gate_usable') is not True
                    or not isinstance(gate.get('quality_gate_pass'), bool)
                    or gate.get('sidedness') != 'one-sided lower'
                    or gate.get('confidence') != gate_plan['confidence'] or gate.get('margin') != margin
                    or gate.get('question_clusters') != validation['question_count']
                    or gate.get('planned_question_count') != gate_plan['question_count']
                    or not _positive_int(gate.get('paired_requests'))
                    or gate['paired_requests'] < validation['question_count']
                    or len(bounds) != 2 or not all(_finite_number(v) for v in bounds)
                    or not _finite_number(gate.get('harmful_probability_upper'))):
                raise ValueError('candidate exact paired development noninferiority evidence is invalid')
            upper = harmful_discordance_upper_bound(gate.get('harmful_questions'), validation['question_count'],
                                                   confidence=gate_plan['confidence'])
            if (gate['quality_gate_pass'] != (upper <= margin)
                    or (execution_criterion == 'exact-noninferiority' and upper > margin)
                    or not math.isclose(bounds[0], -upper, abs_tol=1e-12)
                    or bounds[1] != 1.0
                    or not math.isclose(gate['harmful_probability_upper'], upper, abs_tol=1e-12)):
                raise ValueError('candidate exact paired development noninferiority bound does not pass')
            # Bootstrap intervals remain descriptive. Identical paired outcomes
            # legitimately give a degenerate bootstrap and do not void the exact
            # harmful-discordance bound above.
            bootstrap = validation.get('paired_accuracy_ci', {})
            if bootstrap.get('question_clusters') != validation['question_count']:
                raise ValueError('descriptive paired bootstrap must use the same complete cohort')
        evidence.append(validation)
    for key in ('question_ids_hash', 'sampling_seeds_hash', 'question_count'):
        if evidence[0][key] != evidence[1][key]:
            raise ValueError('reference and candidate independent quality evidence must be paired')
    reference_evidence = next(row for row in selected if row.get('is_default'))['independent_validation']
    candidate_evidence = next(row for row in selected if not row.get('is_default'))['independent_validation']
    reference_accuracy = reference_evidence['metrics']['accuracy']
    candidate_accuracy = candidate_evidence['metrics']['accuracy']
    for summary in (candidate_evidence['paired_accuracy_ci'], candidate_evidence['paired_noninferiority']):
        for key, expected in [('reference_accuracy', reference_accuracy),
                              ('candidate_accuracy', candidate_accuracy)]:
            value = summary.get(key)
            if not _finite_number(value) or not math.isclose(value, expected, abs_tol=1e-10):
                raise ValueError('paired accuracy evidence does not match independent measured accuracies')
    gate = candidate_evidence['paired_noninferiority']
    difference = gate.get('difference_candidate_minus_reference')
    if (not _finite_number(difference)
            or not math.isclose(difference, candidate_accuracy - reference_accuracy, abs_tol=1e-10)
            or difference < -gate['harmful_questions'] / gate['question_clusters'] - 1e-10
            or difference < -margin - 1e-10):
        raise ValueError('paired accuracy difference is inconsistent with the exact harm counts')
    if execution_criterion == 'empirical-paired-accuracy':
        expected_claim = ('statistical_noninferiority_established' if gate['quality_gate_pass']
                          else 'empirical_pass_statistical_noninferiority_inconclusive')
        if candidate_evidence.get('quality_claim_status') != expected_claim:
            raise ValueError('empirical execution gate must explicitly disclose statistical claim status')
    if set(evidence[0]['run_ids']) & set(evidence[1]['run_ids']):
        raise ValueError('reference and candidate must have distinct independent validation runs')
    return copy.deepcopy(dict(frozen))


def run_fixed_compute(freeze_path, *, data_dir, tokenizer, capabilities, environment,
                      executable, output_root, runtime_env=None, seed=724, port=8137, ledger_path=None):
    frozen = validate_frozen_protocol(json.loads(Path(freeze_path).read_text()))
    selected = frozen['selected']
    study = copy.deepcopy(frozen['study'])
    final = study['final']
    allowances = final['reasoning_allowances']
    allowed_seeds = final.get('sampling_seeds', [study['decoding']['seed']])
    if seed not in allowed_seeds:
        raise ValueError('requested sampling seed was not frozen before test access')
    # The collector seeds actual generation from decoding.seed + question index.
    study['decoding']['seed'] = seed
    selected_environments = {row.get('environment_hash') for row in selected}
    if selected_environments != {environment.get('environment_hash')} or None in selected_environments:
        raise ValueError('final execution environment differs from development selections')
    # This read occurs only after the complete development gate and freeze checks.
    official = select_questions(data_dir, 'test')
    ids = frozen['test_question_ids'][:final['question_count']]
    lookup = {q['question_id']: q for q in official}
    if len(lookup) != len(official) or any(qid not in lookup for qid in ids):
        raise ValueError('official test rows do not match the frozen question cohort')
    questions = [lookup[qid] for qid in ids]
    prompts = [prompt_for(q['question'], tokenizer) for q in questions]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    largest = max(lengths)
    if largest + max(allowances) > study['model']['max_model_len']:
        raise ValueError('test prompts exceed frozen context; no test-based adaptation')
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    # Hold all decisions fixed before the first arm. This file never includes gold.
    fixed_reservation=final.get('reserved_wall_seconds')
    cost_note=('Equal nominal whole-device reservations include startup, serving, teardown and allocated idle padding; actual microsecond overshoot retained.'
               if fixed_reservation else 'Common service deadline; whole startup/shutdown reservations can differ. Report the actual frontier.')
    plan = {'freeze_hash': frozen['freeze_hash'], 'environment_hash': environment['environment_hash'],
            'question_ids': ids, 'sampling_seeds_by_question': [seed+i for i in range(len(ids))],
            'prompt_hashes': [digest(prompt) for prompt in prompts],
            'allocated_devices': study['budget']['allocated_devices'], 'service_deadline_s': final['deadline_s'],
            'arm_order': [{'arm': 'default' if row.get('is_default') else 'tuned', 'allowance': allowance}
                          for row in selected for allowance in allowances],
            'reserved_wall_seconds':fixed_reservation,
            'cost_match_note':cost_note}
    write_json(output / f'fixed-compute-plan-seed{seed}.json', plan, immutable=True)
    reference_accuracy = next(row for row in selected if row.get('is_default'))['independent_validation']['metrics']['accuracy']
    arms = []
    for selection in selected:
        arm = 'default' if selection.get('is_default') else 'tuned'
        for allowance in allowances:
            candidate = copy.deepcopy(selection)
            candidate['workload'] = copy.deepcopy(candidate['workload'])
            candidate['scenario_id'] = f'fixed-{arm}-tokens{allowance}'
            candidate['workload'].update(scenario_id=candidate['scenario_id'], output_token_limit=allowance,
                                         partition='holdout', profile='short', input_tokens_max=largest,
                                         input_tokens_mean=sum(lengths)/len(lengths))
            trace = [{'phase': 'measure', 'intended_arrival_s': 0.0} for _ in questions]
            # An all-at-zero finite cohort is a deadline experiment, not a sustained
            # load benchmark. Even never-dispatched questions remain in the cohort.
            protocol = dict(warmup_s=0, ramp_s=0, measurement_s=final['deadline_s'], drain_s=0,
                            timeout_s=final['deadline_s'], max_in_flight=max(512, len(questions)))
            if fixed_reservation:
                protocol.update(reserved_wall_seconds=fixed_reservation,deadline_includes_startup=True,
                                teardown_allowance_s=final.get('teardown_allowance_s',30))
            row = run_trial(candidate=candidate, study=study, capabilities=capabilities, environment=environment,
                            questions=questions, prompts=prompts, output_root=output, executable=executable,
                            tokenizer=tokenizer, seed=seed, phase='validation', runtime_env=runtime_env,
                            run_id=f'fixed-{arm}-{allowance}-seed{seed}', port=port, ledger_path=ledger_path,
                            deadline_mode=True, trace=trace, protocol_override=protocol,
                            reference_accuracy=reference_accuracy)
            path = output / 'runs' / row['run_id'] / 'scored.jsonl'
            if row['status'] == 'ok' and not path.exists():
                raise ValueError('successful final run is missing its immutable scored request evidence')
            if path.exists():
                requests = read_jsonl(path)
            else:
                # A failed collector may still have partial immutable evidence.
                # Preserve actually offered work rather than calling it unoffered.
                raw_path = path.with_name('requests.jsonl')
                if raw_path.exists():
                    from .benchmark import evaluate_collected
                    requests = evaluate_collected(read_jsonl(raw_path), questions, tokenizer)
                else:
                    requests = []
                offered_path = path.with_name('offered.jsonl')
                if offered_path.exists():
                    known = {r['question_id'] for r in requests}
                    for offered in read_jsonl(offered_path):
                        if offered['question_id'] not in known:
                            requests.append({**offered, 'status': 'unanswered', 'finish_reason': None,
                                             'correct': False, 'completion_s': None, 'output_tokens': None})
                            known.add(offered['question_id'])
                if requests:
                    write_json(path.with_name('fixed-compute-recovered-scoring.json'), requests, immutable=True)
            for request in requests:
                index = ids.index(request['question_id']) if request['question_id'] in ids else -1
                if index < 0 or request.get('sampling_seed') not in {None, seed + index}:
                    raise ValueError('scored request does not match the frozen cohort/sampling seed')
                if request.get('status') == 'ok' and request.get('sampling_seed') is None:
                    raise ValueError('completed final request lacks its sampled-seed provenance')
            cost = row['cost']
            expected_cost = study['budget']['allocated_devices'] * cost['wall_seconds']
            if not math.isclose(cost['accelerator_seconds'], expected_cost, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError('final reservation cost does not match the fixed allocated device count')
            effective_deadline=row.get('measurement_duration_s',final['deadline_s'])
            summary = fixed_compute_summary(requests, cohort_ids=ids, deadline_s=effective_deadline,
                                            allocated_accelerators=study['budget']['allocated_devices'],
                                            reservation_wall_seconds=cost['wall_seconds'])
            # Materialize the same seed even when the server failed before dispatch.
            for index, outcome in enumerate(summary['question_outcomes']):
                outcome['sampling_seed'] = seed + index
            arms.append({'arm': arm, 'allowance': allowance, 'run_id': row['run_id'],
                         'execution_status': row['status'], 'summary': summary,
                         'actual_accelerator_seconds': cost['accelerator_seconds'],
                         'reserved_wall_seconds':fixed_reservation,
                         'effective_service_deadline_s':effective_deadline,
                         'cost_match_note': plan['cost_match_note']})
    comparisons = []
    for allowance in allowances:
        reference = next(a for a in arms if a['arm'] == 'default' and a['allowance'] == allowance)
        candidate = next(a for a in arms if a['arm'] == 'tuned' and a['allowance'] == allowance)
        left = reference['summary']['question_outcomes']
        right = candidate['summary']['question_outcomes']
        deadline_left = [dict(r, correct=r['deadline_correct']) for r in left]
        deadline_right = [dict(r, correct=r['deadline_correct']) for r in right]
        comparisons.append({'allowance': allowance,
                            'accuracy_difference': paired_accuracy_ci(left, right, seed=seed),
                            'deadline_accuracy_difference': paired_accuracy_ci(deadline_left, deadline_right, seed=seed),
                            'reference_accelerator_seconds': reference['actual_accelerator_seconds'],
                            'candidate_accelerator_seconds': candidate['actual_accelerator_seconds'],
                            'interpretation':cost_note})
    write_json(output / f'fixed-compute-summary-seed{seed}.json', arms, immutable=True)
    write_json(output / f'fixed-compute-comparisons-seed{seed}.json', comparisons, immutable=True)
    return arms
