#!/usr/bin/env python3
"""Read-only CPU analysis of live continuation evidence; writes derived JSON/plots.

No model, remote service, test answers, or serving runtime is loaded. Immutable
observations and the global ledger are never rewritten. A partially synchronized
snapshot produces an explicitly incomplete report, not inferred missing results.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

METHODS = ('random', 'gp', 'neural')
OVERHEAD = re.compile(r'^cloud-overhead-round(\d+)-(random|gp|neural)$')
TERMINAL_FILE = 'cloud-accounting-final.json'


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_spec(spec):
    original = dict(spec)
    claimed = original.pop('spec_sha256', None)
    if not claimed or claimed != digest(original) or spec.get('heldout_results_seen') is not False:
        raise ValueError('missing, changed, or non-preregistered analysis specification')
    bootstrap = spec['bootstrap']
    if (bootstrap['samples'], bootstrap['seed'], bootstrap['confidence']) != (2000, 724, .95):
        raise ValueError('unexpected preregistered bootstrap settings')


def audit_ledger(events):
    if not events or events[0].get('event') != 'policy':
        raise ValueError('global budget policy is missing')
    begins, finishes = {}, {}
    for event in events[1:]:
        run = event.get('run_id')
        if not run:
            raise ValueError('ledger event lacks run_id')
        if event.get('event') == 'begin':
            if run in begins:
                raise ValueError('duplicate begin event: ' + run)
            devices = event.get('devices')
            if not isinstance(devices, int) or isinstance(devices, bool) or devices < 0:
                raise ValueError('invalid allocated device count')
            number(event.get('max_wall_seconds'), 'reservation bound')
            begins[run] = event
        elif event.get('event') == 'finish':
            if run not in begins or run in finishes:
                raise ValueError('unmatched or duplicate finish event: ' + run)
            cost = number(event.get('accelerator_seconds'), 'ledger accelerator seconds')
            wall = number(event.get('wall_seconds'), 'ledger wall seconds')
            if not math.isclose(cost, wall * begins[run]['devices'], abs_tol=1e-7, rel_tol=1e-10):
                raise ValueError('ledger resource cost mismatch: ' + run)
            finishes[run] = event
        else:
            raise ValueError('unknown ledger event')
    active = sorted(set(begins) - set(finishes))
    total = math.fsum(row['accelerator_seconds'] for row in finishes.values())
    held = math.fsum(begins[run]['devices'] * begins[run]['max_wall_seconds'] for run in active)
    return {'policy':events[0], 'begins':begins, 'finishes':finishes,
            'spent_accelerator_seconds':total, 'held_accelerator_seconds':held,
            'active_run_ids':active}


def search_summary(histories, ledger, plan, *, software_test=False):
    derived = copy.deepcopy(histories)
    finishes = ledger['finishes']
    seen, overhead_used, missing_finishes, missing_overheads = set(), set(), [], []
    curves, methods, configurations, scatter = {}, {}, defaultdict(list), []
    environments = set()
    admission_cap = plan.get('method_budget_s')
    if admission_cap is not None:
        admission_cap = number(admission_cap, 'method base-trial admission cap')
    tagged = {run:event for run,event in finishes.items() if OVERHEAD.fullmatch(run)}
    for run in finishes:
        if run.startswith('cloud-overhead-round') and run not in tagged:
            raise ValueError('malformed method-tagged cloud overhead: ' + run)
    for method in METHODS:
        curve, cumulative, best, best_run = [], 0., None, None
        raw_total, overhead_total, cpu_total, optimizer_wall_total = 0., 0., 0., 0.
        missing_optimizer_wall = 0
        for index, row in enumerate(derived.get(method, [])):
            if row.get('source') != ('synthetic' if software_test else 'measured'):
                raise ValueError('analysis requires explicitly labelled live evidence')
            if row.get('partition') not in {'train', 'validation'} or row.get('workload', {}).get('partition') in {'holdout', 'test'}:
                raise ValueError('held-out outcomes cannot enter search analysis')
            run = row['run_id']
            if run in seen:
                raise ValueError('search run counted more than once: ' + run)
            seen.add(run)
            environments.add(row.get('environment_hash'))
            base = number(row.get('cost', {}).get('accelerator_seconds'), 'trial allocation')
            if run not in finishes:
                missing_finishes.append(run)
            elif not math.isclose(base, finishes[run]['accelerator_seconds'], abs_tol=1e-6, rel_tol=1e-10):
                raise ValueError('raw trial and global ledger cost differ: ' + run)
            tag = f'cloud-overhead-round{index:02d}-{method}'
            extra = tagged.get(tag, {}).get('accelerator_seconds', 0.)
            if tag in tagged:
                if tag in overhead_used:
                    raise ValueError('tagged overhead assigned twice')
                overhead_used.add(tag)
            else:
                missing_overheads.append(tag)
            # Only this detached copy is enriched. The ledger and raw files stay immutable.
            row['cost']['raw_trial_accelerator_seconds'] = base
            row['cost']['assigned_cloud_overhead_seconds'] = extra
            row['cost']['accelerator_seconds'] = base + extra
            raw_total += base
            overhead_total += extra
            cumulative += base + extra
            cpu_total += number(row['cost'].get('optimizer_cpu_seconds', 0), 'optimizer CPU seconds')
            optimizer_wall = row['cost'].get('optimizer_wall_seconds')
            if optimizer_wall is None:
                missing_optimizer_wall += 1
            else:
                optimizer_wall_total += number(optimizer_wall, 'optimizer wall seconds')
            metrics = row.get('metrics', {})
            qag = number(metrics.get('quality_goodput', 0), 'quality goodput')
            feasible = metrics.get('feasible') is True
            if feasible and (best is None or qag > best):
                best, best_run = qag, run
            numerator = metrics.get('quality_slo_completed_in_window')
            curve.append({'trial':index + 1, 'run_id':run, 'config_id':row['config_id'],
                          'raw_trial_accelerator_seconds':base, 'assigned_cloud_overhead_seconds':extra,
                          'overhead_ledger_run_id':tag if tag in tagged else None,
                          'allocation_verified':run in finishes and tag in tagged,
                          'derived_trial_accelerator_seconds':base + extra,
                          'optimizer_cpu_seconds':row['cost'].get('optimizer_cpu_seconds',0),
                          'optimizer_wall_seconds':optimizer_wall,
                          'cumulative_accelerator_seconds':cumulative,
                          'measurement_duration_seconds':row.get('measurement_duration_s', metrics.get('measurement_duration_s')),
                          'window_quality_goodput':qag, 'feasible':feasible, 'run_status':row.get('status'),
                          'best_feasible_goodput':best, 'incumbent_run_id':best_run,
                          'window_correct_per_whole_trial_accelerator_second':
                              numerator / (base + extra) if numerator is not None and base + extra > 0 else None})
            item = {'method':method, 'run_id':run, 'config_id':row['config_id'], 'config':row.get('config', {}),
                    'quality_goodput':qag, 'feasible':feasible, 'status':row.get('status'),
                    'p99_e2e_s':metrics.get('p99_e2e_s'), 'p99_e2e_censored':metrics.get('p99_e2e_censored'),
                    'tail_sample_warning':metrics.get('tail_sample_warning', True)}
            configurations[row['config_id']].append(item)
            scatter.append(item)
        curves[method] = curve
        methods[method] = {'trials':len(curve), 'raw_trial_accelerator_seconds':raw_total,
                           'assigned_cloud_overhead_seconds':overhead_total,
                           'total_attributed_accelerator_seconds':cumulative,
                           'optimizer_cpu_seconds':cpu_total,
                           'optimizer_wall_seconds':optimizer_wall_total if missing_optimizer_wall==0 else None,
                           'known_optimizer_wall_seconds_sum':optimizer_wall_total,
                           'missing_optimizer_wall_time_count':missing_optimizer_wall,
                           'best_feasible_quality_goodput':best,
                           'incumbent_run_id':best_run,
                           'accounting_complete':all(item['allocation_verified'] for item in curve) and bool(curve)}
    if len(environments) > 1 or (seen and None in environments):
        raise ValueError('search analysis cannot mix or omit deployment environments')
    grouped = []
    for config_id, rows in sorted(configurations.items()):
        values = [row['quality_goodput'] for row in rows]
        if len({digest(row['config']) for row in rows}) != 1:
            raise ValueError('same config_id has conflicting configurations')
        grouped.append({'config_id':config_id, 'config':rows[0]['config'], 'count':len(rows),
                        'quality_goodput_mean':math.fsum(values)/len(values),
                        'quality_goodput_min':min(values), 'quality_goodput_max':max(values),
                        'feasible_count':sum(row['feasible'] for row in rows),
                        'methods':sorted({row['method'] for row in rows}),
                        'run_ids':[row['run_id'] for row in rows], 'interval_kind':'observed range, not confidence interval'})
    unmatched = sorted(set(tagged) - overhead_used)
    audit = {'all_tagged_overheads_assigned_once':not unmatched,
             'tagged_overhead_event_count':len(tagged), 'assigned_overhead_event_count':len(overhead_used),
             'assigned_overhead_accelerator_seconds':math.fsum(tagged[run]['accelerator_seconds'] for run in overhead_used),
             'unmatched_tagged_overhead_run_ids':unmatched, 'missing_trial_finish_run_ids':missing_finishes,
             'missing_overhead_run_ids':missing_overheads,
             'complete':not (unmatched or missing_finishes or missing_overheads)}
    expected = plan.get('trials_per_method')
    complete = isinstance(expected, int) and expected > 0 and all(len(curves[m]) == expected for m in METHODS)
    cap_audits={}
    for method,values in methods.items():
        full=values['accounting_complete'] and isinstance(expected,int) and values['trials']==expected
        raw=values['raw_trial_accelerator_seconds'];attributed=values['total_attributed_accelerator_seconds']
        cap_audits[method]={'declared_base_trial_admission_cap_seconds':admission_cap,
            'base_trial_accelerator_seconds':raw,'fully_attributed_accelerator_seconds':attributed,
            'base_trial_within_declared_amount':raw<=admission_cap if admission_cap is not None else None,
            'fully_attributed_within_declared_amount':attributed<=admission_cap if admission_cap is not None else None,
            'base_trial_excess_seconds':max(0.,raw-admission_cap) if admission_cap is not None else None,
            'fully_attributed_excess_seconds':max(0.,attributed-admission_cap) if admission_cap is not None else None,
            'evidence_complete':full,
            'status':('cap_not_configured' if admission_cap is None else 'partial_evidence' if not full else
                      'within_declared_amount' if attributed<=admission_cap else 'exceeds_declared_amount')}
    timing={'optimizer_cpu_seconds_sum':math.fsum(value['optimizer_cpu_seconds'] for value in methods.values()),
        'optimizer_wall_seconds_sum':math.fsum(value['optimizer_wall_seconds'] for value in methods.values())
            if all(value['optimizer_wall_seconds'] is not None for value in methods.values()) else None,
        'scope':'Recorded SearchPolicy.choose process CPU and elapsed wall timers, including fitting/acquisition/artifact work inside that call; not all tokenizer, setup, policy loading or later policy saving CPU time.',
        'resource_note':'Policy CPU/wall timers are descriptive separate clocks, not extra accelerator debits. Reserved GPU wall during this work is already in the uniquely assigned cloud-overhead events. Global ledger optimizer_cpu_seconds may remain zero.'}
    return {'methods':methods, 'curves':curves, 'configuration_repeats':grouped, 'latency_goodput_points':scatter,
            'allocation_audit':audit, 'planned_trials_complete':complete,
            'optimizer_timing':timing,
            'method_cap_audit':{'declared_base_trial_admission_cap_seconds':admission_cap,'methods':cap_audits,
                'affects_completion_gate':False,
                'interpretation':'closed_loop admission uses base trial accelerator charges. Method-tagged cloud overhead is assigned afterward; the same declared amount is audited against fully attributed totals, not asserted as an enforced end-to-end resource cap. Actual excess is retained without changing experiment completion status.'},
            'environment_hash':next(iter(environments)) if environments else None,
            'uncertainty_note':'One search seed per method; no between-search-seed confidence interval. Configuration ranges are descriptive.',
            'cost_note':'Each raw trial plus its one unique method-tagged global-ledger overhead. Shared setup is separate; derived totals are not additional ledger charges.'}, seen, overhead_used


def terminal_accounting(terminal, allocation, ledger):
    reasons = []
    if not terminal:
        return {'verified':False, 'reasons':['terminal cloud accounting not available']}
    if terminal.get('reconciled') is not True:
        reasons.append('terminal artifact does not declare reconciliation')
    if not allocation or terminal.get('allocation_id') != allocation.get('id'):
        reasons.append('terminal allocation identity mismatch')
    basis = terminal.get('duration_basis', 'provider_finish_timestamp')
    upper_bound = basis == 'terminal_observation_upper_bound'
    charged = None
    try:
        created = datetime.fromisoformat(terminal['created_at'])
        if upper_bound:
            if terminal.get('charge_upper_bound') is not True:
                reasons.append('terminal observation accounting must declare a charge upper bound')
            for field in ('provider_finished_at', 'finished_at', 'actual_accelerator_seconds'):
                if field not in terminal or terminal[field] is not None:
                    reasons.append('unknown actual terminal field must be explicitly null: ' + field)
            if terminal.get('terminal_status', {}).get('stage') not in {'CANCELED', 'ERROR', 'COMPLETED'}:
                reasons.append('terminal observation does not confirm a terminal provider state')
            ended = datetime.fromisoformat(terminal['accounting_end_at'])
            observed = datetime.fromisoformat(terminal['terminal_observed_at'])
            if (ended.tzinfo is None or observed.tzinfo is None
                    or ended.utcoffset().total_seconds() != 0 or observed.utcoffset().total_seconds() != 0
                    or observed != ended):
                reasons.append('accounting endpoint and terminal observation must be the same UTC instant')
            charged = number(terminal['charged_accelerator_seconds'], 'terminal charged upper bound')
        else:
            if basis not in {'provider_finish_timestamp', 'provider_finished_at'}:
                reasons.append('unknown terminal duration basis')
            ended = datetime.fromisoformat(terminal['finished_at'])
            charged = number(terminal['actual_accelerator_seconds'], 'terminal allocation')
        if created.tzinfo is None or ended.tzinfo is None:
            raise ValueError('timestamps require explicit timezones')
        duration = (ended - created).total_seconds() * allocation['devices']
        total = number(terminal['global_ledger_spent_accelerator_seconds'], 'terminal ledger total')
        preallocation = number(allocation['preallocation_charged_accelerator_seconds'], 'prior allocation')
        if datetime.fromisoformat(allocation['created_at']) != created or duration < 0:
            reasons.append('terminal timestamps disagree with allocation')
        for left, right, label in [(duration, charged, 'terminal duration'),
                (total, ledger['spent_accelerator_seconds'], 'global ledger total'),
                (preallocation + charged, total, 'prior plus entire cloud allocation')]:
            if not math.isclose(left, right, abs_tol=.01, rel_tol=1e-9):
                reasons.append(label + ' does not reconcile')
    except (KeyError, TypeError, ValueError, AttributeError):
        reasons.append('terminal accounting fields invalid or missing')
    if ledger['active_run_ids']:
        reasons.append('global ledger still has active reservations')
    return {'verified':not reasons, 'reasons':reasons, 'record':terminal,
            'duration_basis':basis, 'charge_upper_bound':upper_bound,
            'charged_accelerator_seconds':charged,
            'actual_accelerator_seconds':None if upper_bound else charged,
            'accounting_note':('The provider confirms a terminal state but supplies no finish timestamp. '
                'Creation through the recorded terminal observation is charged as a conservative upper bound; actual allocation duration is unknown.'
                if upper_bound else 'Allocation duration is reconciled against the provider finish timestamp.'),
            'billing_note':'Estimated charge is not a verified billing invoice.'}


def paired_contrasts(arms, spec):
    """Exact pairing; resample entire question clusters, keeping seed matches."""
    import numpy as np
    allowances = sorted({row['allowance'] for row in arms})
    if len(allowances) != 2:
        raise ValueError('paired analysis requires exactly two allowances')
    maps = {name:{} for name in ('default_short', 'default_long', 'tuned_short', 'tuned_long')}
    for arm in arms:
        name = arm['arm'] + ('_short' if arm['allowance'] == allowances[0] else '_long')
        if name not in maps:
            raise ValueError('unexpected held-out arm name')
        for outcome in arm['summary']['question_outcomes']:
            key = (outcome['question_id'], outcome.get('sampling_seed'))
            if key[1] is None or key in maps[name]:
                raise ValueError('missing or duplicate question/seed held-out pair')
            if not all(isinstance(outcome.get(field), bool) for field in ('correct', 'deadline_correct')):
                raise ValueError('held-out outcomes must explicitly contain Boolean correctness')
            maps[name][key] = outcome
    keys = set(next(iter(maps.values())))
    if not keys or any(set(values) != keys for values in maps.values()):
        raise ValueError('four-arm question/seed cohorts are not identical')
    questions = sorted({key[0] for key in keys})
    grouped = {qid:sorted(key for key in keys if key[0] == qid) for qid in questions}
    rng = np.random.default_rng(spec['bootstrap']['seed'])
    indices = rng.integers(0, len(questions), (spec['bootstrap']['samples'], len(questions)))
    output = []
    for endpoint, field in [('deadline_constrained_accuracy', 'deadline_correct'), ('accuracy', 'correct')]:
        for contrast in spec['contrasts']:
            differences = np.array([sum(sum(weight * int(maps[name][key][field])
                for name,weight in contrast['weights'].items()) for key in grouped[qid]) / len(grouped[qid])
                for qid in questions], dtype=float)
            samples = differences[indices].mean(axis=1)
            output.append({'name':contrast['name'], 'endpoint':endpoint,
                'estimate':float(differences.mean()), 'ci':[float(x) for x in np.quantile(samples, [.025,.975])],
                'question_clusters':len(questions), 'paired_question_seed_rows':len(keys),
                'bootstrap_samples':2000, 'seed':724, 'confidence':.95,
                'method':'paired question-cluster percentile bootstrap',
                'multiplicity':'Descriptive marginal intervals, not simultaneous confidence bounds.'})
    return output


def summarize(root, *, software_test=False):
    root = Path(root)
    out = root/'results'/'continuation'
    spec = read_json(out/'analysis-spec.json')
    verify_spec(spec)
    plan = read_json(out/'search-plan.json', {})
    reference = read_json(out/'default-reference.json')
    ledger = audit_ledger(read_jsonl(root/'results'/'measured'/'budget.jsonl'))
    histories = {method:read_jsonl(out/'search'/method/'observations.jsonl') for method in METHODS}
    search, search_ids, overhead_ids = search_summary(histories, ledger, plan, software_test=software_test)
    categories = defaultdict(list)
    for run, event in ledger['finishes'].items():
        if run in search_ids: category = 'search_raw_trials'
        elif run in overhead_ids: category = 'search_assigned_cloud_overhead'
        elif OVERHEAD.fullmatch(run): category = 'search_unmatched_cloud_overhead'
        elif run == 'cloud-overhead-before-main': category = 'shared_setup_before_main'
        elif run.startswith('cloud-overhead-'): category = 'other_shared_cloud_overhead'
        elif run.startswith('hf-container-start-failure-'): category = 'failed_prior_cloud_allocation'
        elif run.startswith(('pilot-metal-', 'screen-metal-', 'allowance-metal-')): category = 'original_metal_development'
        elif run.startswith('fixed-'): category = 'heldout_trials'
        elif run.startswith('independent-'): category = 'independent_validation'
        else: category = 'other_development_or_allocation'
        categories[category].append({'run_id':run, 'accelerator_seconds':event['accelerator_seconds']})
    category_sums = {name:math.fsum(item['accelerator_seconds'] for item in items) for name,items in categories.items()}
    if not math.isclose(math.fsum(category_sums.values()), ledger['spent_accelerator_seconds'], abs_tol=1e-7):
        raise ValueError('global accounting categories do not sum exactly once to ledger total')
    allocation = read_json(out/'cloud-allocation.json')
    terminal = terminal_accounting(read_json(out/TERMINAL_FILE), allocation, ledger)
    final_dir = out/'final'
    arms, existing, seeds = [], [], []
    for path in sorted(final_dir.glob('fixed-compute-summary-seed*.json')):
        seed = int(re.fullmatch(r'fixed-compute-summary-seed(-?\d+)\.json', path.name)[1])
        seed_arms = read_json(path)
        if not isinstance(seed_arms, list):
            raise ValueError('fixed compute summary must contain an arm list')
        arms.extend(dict(arm, sampling_seed_base=seed) for arm in seed_arms)
        seeds.append(seed)
    for path in sorted(final_dir.glob('fixed-compute-comparisons-seed*.json')):
        existing.append({'file':str(path.relative_to(root)), 'comparisons':read_json(path)})
    final_reasons = []
    freeze = read_json(out/'freeze.json')
    validation_status = read_json(out/'validation-status.json')
    if not freeze:
        final_reasons.append('validated frozen final protocol not available')
    elif not software_test:
        sys.path.insert(0, str(root/'src'))
        from vllm_tuner.fixed_compute import validate_frozen_protocol
        validate_frozen_protocol(freeze)
    if not validation_status or validation_status.get('final_allowed') is not True:
        final_reasons.append('independent validation has not authorized final test')
    elif freeze and validation_status.get('freeze_hash') != freeze.get('freeze_hash'):
        raise ValueError('independent validation status does not match final freeze')
    expected_seeds = freeze['study']['final']['sampling_seeds'] if freeze else [724]
    expected_allowances = freeze['study']['final']['reasoning_allowances'] if freeze else [2048,4096]
    expected_arms = {(seed,arm,tokens) for seed in expected_seeds for arm in ('default','tuned') for tokens in expected_allowances}
    actual_arms = [(arm['sampling_seed_base'], arm['arm'], arm['allowance']) for arm in arms]
    if len(actual_arms) != len(set(actual_arms)):
        raise ValueError('duplicate held-out arm summary')
    if set(actual_arms) != expected_arms:
        final_reasons.append('not all four preregistered arms per seed are available')
    if len({arm['run_id'] for arm in arms}) != len(arms):
        raise ValueError('held-out run reused across arm summaries')
    for arm in arms:
        outcomes = arm['summary'].get('question_outcomes', [])
        ids = [outcome.get('question_id') for outcome in outcomes]
        if not outcomes or len(ids) != len(set(ids)) or len(ids) != arm['summary'].get('cohort_size'):
            raise ValueError('held-out summary lacks its complete unique cohort')
        if freeze:
            count = freeze['study']['final']['question_count']
            frozen_ids = freeze['test_question_ids'][:count]
            if ids != frozen_ids or any(outcome.get('sampling_seed') != arm['sampling_seed_base']+index
                                        for index,outcome in enumerate(outcomes)):
                raise ValueError('held-out cohort or per-question seeds differ from freeze')
        for metric, field in [('accuracy','correct'), ('deadline_constrained_accuracy','deadline_correct')]:
            if not all(isinstance(outcome.get(field), bool) for outcome in outcomes):
                raise ValueError('held-out correctness must be explicit Boolean outcomes')
            observed = math.fsum(outcome[field] for outcome in outcomes) / len(outcomes)
            if not math.isclose(arm['summary'][metric], observed, abs_tol=1e-12):
                raise ValueError('held-out summary accuracy differs from complete cohort')
        event = ledger['finishes'].get(arm['run_id'])
        if event is None:
            final_reasons.append('held-out run lacks terminal ledger charge: ' + arm['run_id'])
        elif not math.isclose(event['accelerator_seconds'], arm['actual_accelerator_seconds'], abs_tol=1e-6):
            raise ValueError('held-out arm cost differs from global ledger')
    contrasts = paired_contrasts(arms, spec) if set(actual_arms) == expected_arms else []
    arm_summaries = []
    for arm in arms:
        compact = copy.deepcopy(arm)
        compact['summary'].pop('question_outcomes', None)
        compact['token_usage_status'] = ('complete' if arm['summary'].get('generated_tokens') is not None
                                         and arm['summary'].get('missing_token_usage_count', 0) == 0 else 'partial_known_lower_bound')
        arm_summaries.append(compact)
    completion_reasons = list(final_reasons)
    if not terminal['verified']: completion_reasons.extend(terminal['reasons'])
    if not search['planned_trials_complete']: completion_reasons.append('planned live comparison incomplete')
    if not search['allocation_audit']['complete']: completion_reasons.append('search allocation reconciliation incomplete')
    if ledger['spent_accelerator_seconds'] > ledger['policy']['total_accelerator_seconds']:
        completion_reasons.append('original accelerator budget exceeded')
    summary = {'schema_version':1, 'source':'software_test' if software_test else 'measured',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'analysis_spec_sha256':spec['spec_sha256'],
        'report_status':'complete' if not completion_reasons else 'incomplete',
        'completion_blockers':list(dict.fromkeys(completion_reasons)), 'search':search,
        'default_reference':reference,
        'accounting':{'global_spent_accelerator_seconds':ledger['spent_accelerator_seconds'],
            'global_held_accelerator_seconds':ledger['held_accelerator_seconds'],
            'original_budget_accelerator_seconds':ledger['policy']['total_accelerator_seconds'],
            'active_run_ids':ledger['active_run_ids'], 'category_accelerator_seconds':category_sums,
            'category_events':dict(categories), 'each_global_finish_counted_once':True,
            'terminal_cloud':terminal,
            'shared_setup_note':'Setup and unassigned allocation are reported separately, never silently added equally to every method.'},
        'independent_validation':{'status':validation_status,
            'evidence':read_json(out/'independent-validation-evidence.json')},
        'heldout':{'four_arms_per_seed_available':set(actual_arms) == expected_arms,
            'arms':arm_summaries, 'existing_comparisons':existing, 'preregistered_paired_contrasts':contrasts,
            'analysis_note':'Full cohort denominator; unknown token usage stays unknown. Known-token sums are lower bounds when usage is missing.'},
        'metric_definitions':{'quality_goodput':'Correct SLO-compliant measurement-window completions divided by measurement seconds.',
            'search_cost_axis':'Actual raw trial allocation plus its uniquely tagged cloud overhead, per method.',
            'whole_trial_efficiency':'Same window-correct numerator divided by whole attributed trial allocation; distinct from window goodput.',
            'configuration_range':'Observed min/max across repeats, not a confidence interval.',
            'latency':'Reported P99 E2E from measured cohort; small-sample warning retained.'}}
    tracked = [out/'analysis-spec.json', out/'search-plan.json', root/'results'/'measured'/'budget.jsonl',
               out/'default-reference.json', out/TERMINAL_FILE, out/'freeze.json',
               out/'validation-status.json', out/'independent-validation-evidence.json']
    tracked += [out/'search'/method/'observations.jsonl' for method in METHODS]
    tracked += sorted(final_dir.glob('*.json'))
    summary['input_sha256'] = {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in tracked if path.exists()}
    summary['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return summary


def plots(summary, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'figure.dpi':150, 'savefig.dpi':180})
    colors = {'random':'#377eb8', 'gp':'#e58320', 'neural':'#36926f'}
    saved = []
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png','svg'):
            path = directory/(name+'.'+suffix)
            fig.savefig(path, bbox_inches='tight')
            saved.append(path.name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.6))
    have = False
    for method in METHODS:
        rows = summary['search']['curves'][method]
        if rows:
            have = True
            ax.step([r['cumulative_accelerator_seconds'] for r in rows],
                    [np.nan if r['best_feasible_goodput'] is None else r['best_feasible_goodput'] for r in rows],
                    where='post', marker='o', markersize=3, color=colors[method], label=method)
    reference = summary.get('default_reference')
    if reference:
        ax.axhline(reference['metrics']['quality_goodput'], linestyle='--', color='#555555', label='development default reference')
    if not have: ax.text(.5,.5,'Live histories not yet synchronized', ha='center', transform=ax.transAxes)
    else: ax.legend(loc='best')
    ax.set(xlabel='Cumulative attributed GPU allocation (seconds)', ylabel='Best feasible window quality goodput (correct/s)',
           title='Live optimizer comparison: trials plus tagged cloud overhead')
    ax.grid(alpha=.2)
    fig.text(.5,-.025,'Shared setup is reported separately. One search seed per method; no confidence bands.', ha='center', fontsize=9)
    save(fig,'live_goodput_vs_allocation')
    fig, ax = plt.subplots(figsize=(8,4.6))
    for method in METHODS:
        curve = summary['search']['curves'][method]
        if curve:
            ax.step([r['trial'] for r in curve],
                    [np.nan if r['best_feasible_goodput'] is None else r['best_feasible_goodput'] for r in curve],
                    where='post', marker='o', markersize=3, color=colors[method], label=method)
    if reference:
        ax.axhline(reference['metrics']['quality_goodput'], linestyle='--', color='#555555', label='development default reference')
    ax.axvline(6.5, color='#888888', linestyle=':', linewidth=1)
    ax.set(xlabel='Completed, independently charged live trials', ylabel='Best feasible window quality goodput (correct/s)',
           title='Shared starting design (1-6), then prospective choices (7-10)', xticks=range(1,11))
    if have: ax.legend(loc='best')
    else: ax.text(.5,.5,'Live histories not yet synchronized',ha='center',transform=ax.transAxes)
    ax.grid(alpha=.2)
    fig.text(.5,-.025,'Trial counts do not include differences in fitting and allocation overhead; see the companion cost curve.',ha='center',fontsize=9)
    save(fig,'live_goodput_vs_trials')
    rows = sorted(summary['search']['configuration_repeats'], key=lambda r:r['quality_goodput_mean'])
    fig, ax = plt.subplots(figsize=(8,max(4, .33*len(rows)+1.6)))
    if rows:
        mean = np.array([r['quality_goodput_mean'] for r in rows])
        ax.errorbar(mean, range(len(rows)), xerr=[mean-np.array([r['quality_goodput_min'] for r in rows]),
                    np.array([r['quality_goodput_max'] for r in rows])-mean], fmt='o', capsize=3, color='#555a91')
        labels = [f"batch {r['config'].get('max_num_batched_tokens') or 'default'}, seq {r['config'].get('max_num_seqs') or 'default'} (n={r['count']})" for r in rows]
        ax.set_yticks(range(len(rows)), labels)
    else: ax.text(.5,.5,'Configuration observations pending',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Window quality goodput (correct/s)', title='Configuration means and observed ranges — not confidence intervals')
    ax.grid(axis='x',alpha=.2)
    save(fig,'configuration_goodput_ranges')
    fig, ax = plt.subplots(figsize=(7.6,4.8))
    omitted=0
    for method in METHODS:
        pts = [p for p in summary['search']['latency_goodput_points'] if p['method']==method]
        for feasible, marker in [(True,'o'),(False,'x')]:
            selected=[p for p in pts if p['feasible']==feasible and isinstance(p['p99_e2e_s'],(int,float)) and math.isfinite(p['p99_e2e_s'])]
            if selected:ax.scatter([p['p99_e2e_s'] for p in selected],[p['quality_goodput'] for p in selected],
                 marker=marker,color=colors[method],alpha=.75,label=method+(' feasible' if feasible else ' infeasible'))
        omitted+=sum(p['p99_e2e_s'] is None or not isinstance(p['p99_e2e_s'],(int,float)) or not math.isfinite(p['p99_e2e_s']) for p in pts)
    if ax.collections:ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Measured latency points pending',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Observed P99 end-to-end latency (seconds)',ylabel='Window quality goodput (correct/s)',
           title='Latency and quality goodput; small-sample tails are descriptive')
    fig.text(.5,-.02,f'{omitted} undefined/censored latency points omitted from coordinates; retained in JSON.',ha='center',fontsize=9)
    ax.grid(alpha=.2)
    save(fig,'latency_goodput')
    fig, (left,right) = plt.subplots(1,2,figsize=(11,4.8))
    arms = summary['heldout']['arms']
    if summary['heldout']['four_arms_per_seed_available']:
        arms=sorted(arms,key=lambda a:(a['sampling_seed_base'],a['arm'],a['allowance']))
        x=np.arange(len(arms))
        labels=[f"{a['arm']}\n{a['allowance']} tokens" for a in arms]
        left.bar(x-.18,[a['summary']['deadline_constrained_accuracy'] for a in arms],.36,label='deadline-constrained')
        left.bar(x+.18,[a['summary']['accuracy'] for a in arms],.36,label='full-cohort accuracy',color='#93b4c9')
        left.set_ylim(0,1);left.legend(fontsize=8)
        for i,a in enumerate(arms):
            known=a['summary'].get('known_output_tokens_sum',0)
            partial=a['token_usage_status']!='complete'
            right.bar(i,known,color='#36926f',hatch='///' if partial else None,alpha=.8)
            note=f"≥{known:,}\n{a['summary'].get('missing_token_usage_count',0)} usage unknown" if partial else f'{known:,}'
            right.annotate(note,(i,known),ha='center',xytext=(0,4),textcoords='offset points',fontsize=8)
        right.margins(y=.25)
        for ax in (left,right):ax.set_xticks(x,labels)
    else:
        for ax in (left,right):ax.text(.5,.5,'Four held-out arm results pending\nNo final values inferred',ha='center',va='center',transform=ax.transAxes)
    left.set(title='Held-out full-cohort accuracy',ylabel='Fraction correct')
    right.set(title='Generated output tokens: known usage',ylabel='Tokens (hatched bars are lower bounds)')
    fig.text(.5,-.025,'Actual allocations and paired contrast intervals are in summary.json. Unknown usage is never replaced with zero.',ha='center',fontsize=9)
    save(fig,'heldout_accuracy_and_tokens')
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    summary=summarize(args.root)
    destination=args.root/'results'/'continuation'/'analysis'
    destination.mkdir(parents=True,exist_ok=True)
    summary['plot_files']=[] if args.no_plots else plots(summary,destination)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'report_status':summary['report_status'],'summary':str(destination/'summary.json'),
                      'blockers':summary['completion_blockers'],'plots':summary['plot_files']}))


if __name__=='__main__':
    main()
