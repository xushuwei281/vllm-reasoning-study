"""One serial accelerator trial queue, with explicit failed runs and bounded cost."""
from __future__ import annotations
import asyncio
import copy
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
from .accounting import BudgetLedger, RunJournal, BudgetExceeded
from .benchmark import collect, evaluate_collected,validate_trace
from .config import validate_config, build_serve_command
from .data import digest, prompt_for, write_json, write_jsonl, select_questions
from .evaluation import summarize_requests, queue_trace
from .lifecycle import ServerProcess


def candidates_for(study, workload):
    keys=list(study['space']); candidates=[]
    for values in itertools.product(*(study['space'][k] for k in keys)):
        config={**study['fixed_config'],**dict(zip(keys,values))}
        candidates.append({'config_id':digest(config)[:12],'scenario_id':workload['scenario_id'],
                           'config':config,'workload':copy.deepcopy(workload),'known_valid':True})
    return candidates


def prepare_workload(study, workload, data_dir, tokenizer, partition='profiling', limit=None):
    questions=select_questions(data_dir,partition)
    if limit: questions=questions[:limit]
    profiles=[('long_context_stress' if i%2 else 'short') if workload['profile']=='mixed' else workload['profile'] for i in range(len(questions))]
    prompts=[prompt_for(q['question'],tokenizer,profile) for q,profile in zip(questions,profiles)]
    lengths=[len(tokenizer.encode(p,add_special_tokens=False)) for p in prompts]
    descriptor={**workload,'input_tokens_mean':float(np.mean(lengths)),
                'input_tokens_p95':float(np.percentile(lengths,95)), 'input_tokens_max':max(lengths)}
    if max(lengths)+workload['output_token_limit']>study['model']['max_model_len']:
        raise ValueError('prompt + output allowance exceeds context; no silent truncation')
    return questions,prompts,descriptor


def run_trial(*, candidate, study, capabilities, environment, questions, prompts,
              output_root, executable, tokenizer=None, repetition=0, seed=0,
              reference_accuracy=None, phase='search', runtime_env=None,
              run_id=None, source='measured', protocol_override=None, port=8000,
              deadline_mode=False, trace=None, ledger_path=None):
    # Validate client-side inputs before reserving hardware or creating a run.
    if source not in {'measured', 'synthetic'}:
        raise ValueError('source must explicitly distinguish measured and synthetic')
    if not questions or len(questions) != len(prompts):
        raise ValueError('question/prompt mismatch')
    if not isinstance(repetition, int) or isinstance(repetition, bool) or repetition < 0:
        raise ValueError('repetition must be a nonnegative integer')
    if phase not in {'search', 'calibration', 'validation', 'optimizer'}:
        raise ValueError('unknown budget phase')
    if reference_accuracy is not None and not 0 <= reference_accuracy <= 1:
        raise ValueError('reference_accuracy must be in [0, 1]')
    root=Path(output_root)
    run_id=run_id or f'{candidate["scenario_id"]}-{candidate["config_id"]}-r{repetition}-s{seed}'
    journal=RunJournal(root/'observations.jsonl')
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_id must be a single path component')
    effective_ledger_path = Path(ledger_path or root/'budget.jsonl').resolve()
    directory=root/'runs'/run_id
    protocol={**study['benchmark'],'arrival_rate':candidate['workload']['arrival_rate'],**(protocol_override or {})}
    validate_trace(trace or [],protocol)
    reservation = trial_reservation_bound(study, protocol_override)
    if protocol.get('deadline_includes_startup') and not deadline_mode:
        raise ValueError('whole-reservation deadline requires fixed-cohort mode')
    import math
    if trace is None and (not isinstance(protocol.get('arrival_rate'), (int, float))
                          or not math.isfinite(protocol['arrival_rate']) or protocol['arrival_rate'] <= 0):
        raise ValueError('open-loop arrival_rate must be finite and positive')
    if not isinstance(protocol['max_in_flight'], int) or isinstance(protocol['max_in_flight'], bool):
        raise ValueError('max_in_flight must be a positive integer')
    interval = protocol.get('telemetry_interval_s', 1)
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError('telemetry_interval_s must be finite and positive')
    if deadline_mode and (trace is None or len(trace) != len(questions)):
        raise ValueError('fixed cohort requires an explicit arrival per question before reservation')
    # Hash rather than publish the effective environment: it may contain secrets,
    # while backend-affecting variables must still invalidate stale reuse.
    execution_controls = {'repetition': repetition, 'seed': seed, 'reference_accuracy': reference_accuracy,
                          'phase': phase, 'deadline_mode': deadline_mode, 'source': source}
    fingerprint=digest({'study':study,'candidate':candidate,'environment':environment['environment_hash'],
                       'questions':questions,'prompts':prompts,'protocol':protocol,'trace':trace,
                       'controls':execution_controls, 'capabilities':capabilities,
                       'executable':str(Path(executable).resolve()), 'port':port,
                       'ledger_path':str(effective_ledger_path),
                       'evaluator_hash':digest(Path(__file__).with_name('evaluation.py').read_text()),
                       'runtime_environment_hash':digest(dict(os.environ) if runtime_env is None else dict(runtime_env))})
    if journal.contains(run_id):
        existing=next(r for r in journal.records() if r['run_id']==run_id)
        if existing.get('request_fingerprint')!=fingerprint:raise ValueError('existing run ID has different or unaudited inputs; refusing stale reuse')
        return existing
    if directory.exists(): raise RuntimeError('incomplete run exists; preserve it, reconcile ledger, retry under a new ID')
    ledger=BudgetLedger(effective_ledger_path,study['budget']['total_accelerator_seconds'],study['budget']['validation_fraction'])
    config=candidate['config']; devices=study['budget']['allocated_devices']
    if source == 'measured' and (not isinstance(devices, int) or isinstance(devices, bool) or devices < 1):
        raise ValueError('measured accelerator trials require a positive allocated device count')
    row={**copy.deepcopy(candidate),'run_id':run_id,'repetition':repetition,'source':source,
         'partition':'holdout' if candidate['workload'].get('partition')=='holdout' else 'train',
         'environment_hash':environment['environment_hash'],'status':'pending','seed':seed,'request_fingerprint':fingerprint}
    errors=validate_config(config,capabilities,candidate['workload'])
    if devices > capabilities.get('device_count', 0):
        errors.append('allocated deployment fleet exceeds available devices')
    if errors:
        directory.mkdir(parents=True)
        row.update(status='known_invalid',failure='; '.join(errors),metrics={'quality_goodput':0.0,'p99_e2e_s':None,'accuracy':0.0,'feasible':False},cost={'accelerator_seconds':0.0,'wall_seconds':0.0,'optimizer_cpu_seconds':0.0})
        journal.append(row);write_json(directory/'observation.json',row,immutable=True);return row
    command=build_serve_command(config,study['model'],port=port,executable=executable)
    startup=reservation.get('startup_timeout_s',study.get('startup_timeout_s',240)); shutdown=15
    upper=reservation['wall_seconds']
    if not ledger.begin(run_id,devices,upper,phase): raise RuntimeError('reservation already exists without result; recover first')
    directory.mkdir(parents=True)
    write_json(directory/'provenance.json',{'command':command,'study_hash':digest(study),'candidate':candidate,
                                           'environment_hash':environment['environment_hash'],'source':source,
                                           'execution_controls':execution_controls, 'request_fingerprint':fingerprint},immutable=True)
    server=ServerProcess(command,directory/'server.log',devices,startup,shutdown,
                         health_url=f'http://127.0.0.1:{port}/health',env=runtime_env)
    reserved = protocol.get('reserved_wall_seconds')
    execution_protocol = copy.deepcopy(protocol)
    try:
        server.start()
        if protocol.get('deadline_includes_startup'):
            # Leave a preregistered teardown allowance. Startup consumes the same
            # finite arm allocation as serving; it is never free compute.
            available = (reserved - (time.monotonic()-server.started_at)
                         - protocol.get('teardown_allowance_s', 30)
                         - protocol['warmup_s'] - protocol['ramp_s'])
            if available <= 0: raise TimeoutError('startup exhausted fixed arm allocation')
            execution_protocol.update(measurement_s=available, timeout_s=available)
        write_json(directory/'execution-protocol.json', execution_protocol, immutable=True)
        rows,memory,info=asyncio.run(collect(f'http://127.0.0.1:{port}',study['model']['name'],questions,prompts,
            {**study['decoding'],'max_tokens':candidate['workload']['output_token_limit']},execution_protocol,directory,
            source=source,seed=seed,deadline_mode=deadline_mode,trace=trace))
        # Release accelerator before CPU parsing/tokenization/optimizer work.
        server.stop()
        scored=evaluate_collected(rows,questions,tokenizer)
        write_jsonl(directory/'scored.jsonl',scored);write_jsonl(directory/'queue.jsonl',queue_trace(scored))
        mapped=[{'time_s':s['time_s'],'used_bytes':s.get('device_memory_bytes')} for s in memory]
        kwargs=dict(window_start_s=info['window_start_s'],window_end_s=info['window_end_s'],slos=study['slos'],
                    allocated_accelerators=devices,reference_accuracy=reference_accuracy,**study['constraints'],memory_samples=mapped)
        metrics=summarize_requests(scored,**kwargs)
        if candidate.get('is_default') and reference_accuracy is None:
            kwargs['reference_accuracy']=metrics['accuracy'];metrics=summarize_requests(scored,**kwargs)
            metrics['quality_constraint_status']='reference_self; candidate comparisons require development baseline'
        row.update(status='ok',metrics=metrics,measurement_duration_s=execution_protocol['measurement_s'],request_count=len(rows),trace_hash=info['trace_hash'],trace_id=info['trace_hash'])
    except (Exception,KeyboardInterrupt) as exc:
        row.update(status='failed',failure=f'{type(exc).__name__}: {exc}',
                   metrics={'quality_goodput':0.0,'p99_e2e_s':None,'p99_e2e_censored':True,'accuracy':0.0,'feasible':False,'offered':None,'completed':0})
        if isinstance(exc,KeyboardInterrupt): row['interrupted']=True
    finally:
        server.stop();row['cost']=server.cost()
        if reserved is not None and server.started_at is not None:
            stopped_wall=row['cost']['wall_seconds']
            padding_start = time.monotonic()
            # The cloud device remains allocated and unavailable to any other
            # trial during padding. Keep actual overshoot, never clip charges.
            while True:
                remaining = reserved - (time.monotonic()-server.started_at)
                if remaining <= 0: break
                time.sleep(min(1.0, remaining))
            elapsed=time.monotonic()-server.started_at
            row['cost'].update(wall_seconds=elapsed, accelerator_seconds=elapsed*devices,
                               reserved_slot_seconds=reserved, server_lifecycle_seconds=stopped_wall,
                               allocated_idle_padding_seconds=max(0, elapsed-(padding_start-server.started_at)),
                               allocated_after_shutdown_seconds=max(0, elapsed-stopped_wall),
                               effective_startup_timeout_seconds=startup,
                               admission_wall_seconds=upper,
                               admission_bound_overshoot_seconds=max(0, elapsed-upper),
                               reservation_timing_policy=('whole_reservation_deadline' if protocol.get('deadline_includes_startup')
                                                          else 'minimum_allocated_slot'),
                               slot_overshoot_seconds=max(0, elapsed-reserved))
        ledger.finish(run_id,row['cost']['wall_seconds'],status=row['status'])
        write_json(directory/'observation.json',row,immutable=True);journal.append(row)
    if row.get('interrupted'): raise KeyboardInterrupt
    return row


def trial_reservation_bound(study, protocol_override=None):
    """One source for trial admission and per-method conservative budget gates.

    Includes both graceful and forced shutdown waits plus readiness/dispatch
    overhead. It is an admission estimate, not a guarantee against OS failure;
    actual overruns must remain charged and stop further search admission.
    """
    import math
    protocol = {**study['benchmark'], **(protocol_override or {})}
    startup = study.get('startup_timeout_s', 240)
    shutdown = 15.0
    phases = [protocol[name] for name in ('warmup_s', 'ramp_s', 'measurement_s', 'drain_s')]
    devices = study['budget']['allocated_devices']
    if not isinstance(devices, int) or isinstance(devices, bool) or devices < 0:
        raise ValueError('allocated_devices must be a nonnegative integer')
    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not finite_number(startup) or startup <= 0 or any(not finite_number(x) or x < 0 for x in phases):
        raise ValueError('finite positive startup and nonnegative phase durations required')
    whole_deadline = protocol.get('deadline_includes_startup', False)
    if not isinstance(whole_deadline, bool):
        raise ValueError('deadline_includes_startup must be boolean')
    slot = protocol.get('reserved_wall_seconds')
    if whole_deadline and slot is None:
        raise ValueError('whole-reservation deadline requires a reserved slot')
    if slot is not None:
        if not finite_number(slot) or slot <= 0:
            raise ValueError('reserved slot must be finite and positive')
        teardown = protocol.get('teardown_allowance_s', 2 * shutdown)
        overhead = protocol.get('reservation_overhead_allowance_s', 15.0)
        if not finite_number(teardown) or teardown < 2 * shutdown:
            raise ValueError('teardown allowance must cover both 15-second shutdown waits')
        if not finite_number(overhead) or overhead < 0:
            raise ValueError('reservation overhead allowance must be finite and nonnegative')
        startup_room = slot - teardown - protocol['warmup_s'] - protocol['ramp_s']
        if startup_room <= 0:
            raise ValueError('reserved slot must exceed teardown, warmup, and ramp allowances')
        startup = min(startup, startup_room)
        # A padded slot alone is a minimum allocation, not a cutoff. The usual
        # collection/drain can exceed it. Whole-deadline mode instead shortens
        # collection after observing startup; keep overhead outside its target.
        wall = (slot if whole_deadline else max(slot, startup + sum(phases) + 2 * shutdown)) + overhead
        return {'wall_seconds':float(wall), 'accelerator_seconds':devices*wall,
                'reserved_slot_seconds':float(slot), 'startup_timeout_s':float(startup),
                'reservation_is_hard_upper_bound':False}
    wall = startup + 2 * shutdown + sum(phases) + 15.0
    return {'wall_seconds': wall, 'accelerator_seconds': devices * wall}


def _validate_search_result(result, candidate):
    import math
    if result.get('source') != 'measured' or result.get('partition') not in {'train', 'validation'}:
        raise ValueError('search accepts measured development rows only')
    if result.get('workload', {}).get('partition') in {'holdout', 'test'}:
        raise ValueError('held-out workload cannot enter search')
    if not result.get('run_id') or result.get('status') not in {'ok', 'failed', 'known_invalid', 'timeout', 'error'}:
        raise ValueError('runner must return a completed immutable run record')
    if (result.get('config_id'), result.get('scenario_id')) != (candidate['config_id'], candidate['scenario_id']):
        raise ValueError('runner returned a different candidate')
    if result.get('config') != candidate['config']:
        raise ValueError('runner changed the frozen configuration')
    if not result.get('environment_hash'):
        raise ValueError('measured result requires an environment hash')
    metrics = result.get('metrics', {})
    value = metrics.get('quality_goodput')
    if not isinstance(metrics.get('feasible'), bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('completed result requires finite nonnegative goodput and explicit measured feasibility')
    for name in ('accelerator_seconds', 'wall_seconds'):
        value = result.get('cost', {}).get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('runner costs must be finite and nonnegative')


def _attach_optimizer_cost(result, decision):
    import math
    result = copy.deepcopy(result)
    result['optimizer_decision'] = copy.deepcopy(decision)
    for name in ('optimizer_cpu_seconds', 'optimizer_wall_seconds'):
        cost = float(result['cost'].get(name, 0)) + float(decision.get(name, 0))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('optimizer costs must be finite and nonnegative')
        result['cost'][name] = cost
    return result


def _save_incumbent(directory, observed):
    from .search import measured_incumbent
    write_json(directory / 'incumbent.json', {
        'best_measured': measured_incumbent(observed),
        'spent_accelerator_seconds': sum(r['cost']['accelerator_seconds'] for r in observed),
        'optimizer_cpu_seconds': sum(r['cost'].get('optimizer_cpu_seconds', 0) for r in observed),
        'optimizer_wall_seconds': sum(r['cost'].get('optimizer_wall_seconds', 0) for r in observed),
        'needs_independent_validation': True,
    })


def recover_pending(output_dir, live_journal_path, run_id):
    """Explicitly reconcile one proposal against an already completed live run.

    Never launches a server, invents a failure cost, or clears an unmatched
    reservation. The caller first finishes/reconciles the live resource ledger.
    run_trial appends its immutable live row only after ledger.finish, making
    that row the evidence required here. A second call is idempotent.
    """
    directory = Path(output_dir)
    journal = RunJournal(directory / 'observations.jsonl')
    observed = journal.records()
    live = [r for r in RunJournal(live_journal_path).records() if r['run_id'] == run_id]
    if len(live) != 1:
        raise ValueError('recovery requires exactly one completed record in the immutable live journal')
    pending_path = directory / 'pending.json'
    existing = next((r for r in observed if r['run_id'] == run_id), None)
    if not pending_path.exists():
        if existing is not None:
            return existing
        raise RuntimeError('no pending proposal to reconcile')
    pending = json.loads(pending_path.read_text())
    _validate_search_result(live[0], pending['candidate'])
    result = _attach_optimizer_cost(live[0], pending['decision'])
    if observed and any(r['environment_hash'] != result['environment_hash'] for r in observed):
        raise ValueError('recovery cannot mix deployment environments')
    index = pending['index']
    if existing is not None:
        if len(observed) != index + 1 or observed[-1] != result:
            raise ValueError('pending proposal conflicts with already recorded search history')
    else:
        if len(observed) != index:
            raise ValueError('pending index differs from immutable search history')
        if any((r['config_id'], r['scenario_id']) == (result['config_id'], result['scenario_id']) for r in observed):
            raise ValueError('recovery would duplicate a searched candidate')
        journal.append(result)
        observed.append(result)
    _save_incumbent(directory, observed)
    write_json(directory / f'recovery-{index}-{digest(run_id)[:12]}.json', {
        'run_id': run_id, 'candidate_id': result['config_id'], 'action': 'matched_completed_live_record',
        'live_journal': str(Path(live_journal_path).resolve()), 'accelerator_seconds': result['cost']['accelerator_seconds'],
    }, immutable=True)
    pending_path.unlink()
    contract = json.loads((directory / 'search-contract.json').read_text())
    if result['cost']['accelerator_seconds'] > contract['maximum_trial_cost_s']:
        raise RuntimeError('recovered cost exceeds declared bound; cost retained, search remains stopped')
    return result


def closed_loop(candidates, runner, output_dir, *, method='neural', seed=0, initial_points=6,
                accelerator_budget_s=5400, maximum_trial_cost_s=600, neural_epochs=200, max_trials=None):
    """runner(candidate,index) returns a charged real development observation.

    The runner releases inference accelerators between trials. CPU optimizer
    time is recorded separately; callers retaining externally reserved devices
    must additionally charge that reservation time in the live resource ledger.
    Shared starts are independently charged for each live method. A pending
    proposal requires explicit recover_pending, never a silent retry.
    """
    import math
    from .search import SearchPolicy, initial_design
    if len({c['scenario_id'] for c in candidates}) != 1:
        raise ValueError('one static scenario per search')
    if len({(c['config_id'], c['scenario_id']) for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate in frozen search space')
    if any('metrics' in c or 'cost' in c for c in candidates):
        raise ValueError('candidate descriptors must not contain unobserved outcomes or costs')
    if any(c.get('partition') in {'holdout', 'test'} or c.get('workload', {}).get('partition') in {'holdout', 'test'} for c in candidates):
        raise ValueError('held-out workload cannot enter search')
    if not math.isfinite(maximum_trial_cost_s) or maximum_trial_cost_s <= 0 or not math.isfinite(accelerator_budget_s) or accelerator_budget_s < 0:
        raise ValueError('invalid budget bounds')
    if not isinstance(initial_points, int) or initial_points < 0:
        raise ValueError('initial_points must be a nonnegative integer')
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    contract = {'candidate_hash': digest(candidates), 'method': method, 'seed': seed, 'initial_points': initial_points,
                'accelerator_budget_s': accelerator_budget_s, 'maximum_trial_cost_s': maximum_trial_cost_s, 'neural_epochs': neural_epochs}
    write_json(directory / 'search-contract.json', contract, immutable=True)
    if (directory / 'pending.json').exists():
        raise RuntimeError('pending proposal requires explicit recovery against immutable live journal before resume')
    journal = RunJournal(directory / 'observations.jsonl'); observed = journal.records()
    by_key = {(c['config_id'], c['scenario_id']): c for c in candidates}
    seen = set()
    for row in observed:
        key = row['config_id'], row['scenario_id']
        if key not in by_key:
            raise ValueError('observations outside frozen search space')
        if key in seen:
            raise ValueError('duplicate searched candidate in immutable history')
        _validate_search_result(row, by_key[key]); seen.add(key)
        if row['cost']['accelerator_seconds'] > maximum_trial_cost_s:
            raise RuntimeError('previous trial exceeded declared bound; cost retained, search remains stopped')
    if len({r['environment_hash'] for r in observed}) > 1:
        raise ValueError('search cannot mix deployment environments')
    state_path = directory / 'policy.json'
    expected = SearchPolicy(method, seed, neural_epochs=neural_epochs)
    if observed and not state_path.exists():
        raise RuntimeError('search history exists without policy checkpoint; explicit recovery required')
    policy = SearchPolicy.load(state_path) if state_path.exists() else expected
    policy.artifact_directory = directory / 'models'
    if policy.settings != expected.settings:
        raise ValueError('policy checkpoint settings differ from frozen search contract')
    common = initial_design(candidates, initial_points, seed) if method != 'default' else []
    while True:
        if max_trials is not None and len(observed) >= max_trials:
            break
        spent = sum(r['cost']['accelerator_seconds'] for r in observed)
        if spent + maximum_trial_cost_s > accelerator_budget_s:
            break
        from_initial = len(observed) < len(common)
        candidate = common[len(observed)] if from_initial else policy.choose(candidates, observed)
        if candidate is None:
            break
        decision = {'method_used': 'shared_initial_design', 'optimizer_cpu_seconds': 0., 'optimizer_wall_seconds': 0.} if from_initial else copy.deepcopy(policy.last_decision)
        policy.save(state_path)
        write_json(directory / 'pending.json', {'candidate': candidate, 'index': len(observed), 'spent': spent, 'decision': decision})
        raw_result = runner(candidate, len(observed))
        _validate_search_result(raw_result, candidate)
        if observed and any(r['environment_hash'] != raw_result['environment_hash'] for r in observed):
            raise ValueError('search cannot mix deployment environments')
        if any(r['run_id'] == raw_result['run_id'] for r in observed):
            raise ValueError('runner returned a run ID already charged for another proposal')
        result = _attach_optimizer_cost(raw_result, decision)
        journal.append(result); observed.append(result)
        _save_incumbent(directory, observed)
        (directory / 'pending.json').unlink()
        if result['cost']['accelerator_seconds'] > maximum_trial_cost_s:
            raise RuntimeError('actual trial exceeded declared bound; cost retained, search stopped')
    return observed
