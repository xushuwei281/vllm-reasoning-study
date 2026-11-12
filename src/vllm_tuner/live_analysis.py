"""Independent audit/summary of live optimizer histories; no synthetic replay."""
from __future__ import annotations
import math
from typing import Mapping,Sequence,Any
from .search import measured_incumbent


def validate_planned_trace_hashes(expected_trace_hashes: Sequence[str], required_count: int = 0) -> tuple[str, ...]:
    """Validate actual schedule digests before launching a matched-trace campaign."""
    if isinstance(expected_trace_hashes, (str, bytes)):
        raise ValueError('planned trace hashes must be a sequence of distinct nonempty strings')
    hashes = tuple(expected_trace_hashes)
    if not hashes or any(not isinstance(value, str) or not value.strip() for value in hashes):
        raise ValueError('planned trace hashes must be distinct nonempty strings')
    if len(set(hashes)) != len(hashes):
        raise ValueError('planned trace hashes must be distinct within each optimizer history')
    if len(hashes) < required_count:
        raise ValueError('planned trace hashes do not cover every observed trial index')
    return hashes


def audit_planned_traces(histories: Mapping[str, Sequence[Mapping[str, Any]]],
                        expected_trace_hashes: Sequence[str]) -> dict[str, Any]:
    """Verify executed traces against the plan, retaining unverified failed runs.

    Failed startup/collection may have no measured trace identity. Such rows
    remain in the cost/learning history; their intended trace is not fabricated
    into an actual hash. Available hashes must match even for failed trials.
    """
    hashes = validate_planned_trace_hashes(expected_trace_hashes,
                                          max((len(rows) for rows in histories.values()), default=0))
    methods = {}
    failure_states = {'failed', 'known_invalid', 'error', 'timeout', 'oom', 'interrupted', 'startup_failed'}
    for method, records in histories.items():
        trials = []; seen_trace_ids = set()
        for index, row in enumerate(records):
            actual = row.get('trace_hash')
            trace_id = row.get('trace_id')
            if trace_id:
                if trace_id in seen_trace_ids:
                    raise ValueError(f'{method}: repeated trace_id links distinct planned trials')
                seen_trace_ids.add(trace_id)
            if actual:
                if actual != hashes[index]:
                    raise ValueError(f'{method} trial {index + 1}: actual trace hash differs from the frozen plan')
                status = 'matched'
            else:
                if row.get('status') not in failure_states or row.get('metrics', {}).get('feasible', False):
                    raise ValueError(f'{method} trial {index + 1}: successful or unclassified row lacks an actual trace hash')
                status = 'unverified_failure'
            trials.append({'trial': index + 1, 'run_id': row['run_id'], 'run_status': row.get('status'),
                           'expected_trace_hash': hashes[index], 'actual_trace_hash': actual or None,
                           'trace_id': trace_id, 'audit_status': status})
        methods[method] = {'matched_trials': sum(t['audit_status'] == 'matched' for t in trials),
                           'unverified_failed_trials': sum(t['audit_status'] == 'unverified_failure' for t in trials),
                           'trials': trials}
    return {'enabled': True, 'expected_trace_hashes': list(hashes), 'methods': methods,
            'note': 'Actual hashes are checked against the same per-index plan across methods. Failed rows without an actual hash remain charged and are explicitly unverified.'}


def summarize_live_search(histories: Mapping[str, Sequence[Mapping[str,Any]]], *,
                          common_initial_count: int = 6,
                          permitted_shared_run_ids: Sequence[str] = (),
                          checkpoint_count: int = 11,
                          expected_trace_hashes: Sequence[str] | None = None) -> dict[str,Any]:
    """Compare incumbents only after their observations were actually paid for.

    Common initial records may be physically shared only when their run IDs are
    declared explicitly. Such reuse is separated from actual physical costs;
    repeated logical debits must not be called additional accelerator usage.
    With independent live starts the permitted list should be empty.
    Optional planned trace hashes validate matched actual schedules by trial
    index. Missing hashes on explicit failed rows stay visible and charged.
    """
    methods=tuple(histories)
    if not methods or checkpoint_count<2 or common_initial_count<0:
        raise ValueError('nonempty histories and valid common-design/checkpoint sizes required')
    trace_audit = (audit_planned_traces(histories, expected_trace_hashes)
                   if expected_trace_hashes is not None else
                   {'enabled': False, 'note': 'No expected trace hashes supplied; matching arrival schedules was not audited.'})
    seen={};environment=set();scenario=set();shared=set(permitted_shared_run_ids)
    initial_designs=[];curves={};summaries={}
    for method,records in histories.items():
        if not records:raise ValueError(f'{method}: no live observations')
        local_ids=set();running=[];cost=0.;cpu=0.;wall=0.;curve=[]
        initial_designs.append([(r['config_id'],r['scenario_id']) for r in records[:common_initial_count]])
        for index,row in enumerate(records):
            if row.get('source')!='measured' or row.get('partition') not in {'train','validation'}:
                raise ValueError('live search analysis accepts measured development observations only')
            if row.get('workload',{}).get('partition') in {'test','holdout'}:
                raise ValueError('held-out outcomes cannot enter optimizer learning curves')
            run_id=row['run_id']
            if run_id in local_ids:raise ValueError('duplicate run in one optimizer history')
            if run_id in seen and run_id not in shared:
                raise ValueError('cross-method run reuse must be declared as common physical evidence')
            if run_id in seen and dict(row)!=seen[run_id]:
                # Optimizer decision annotation may differ. The paid observation
                # itself must still be identical across allowed shared records.
                a={k:v for k,v in row.items() if k!='optimizer_decision'}
                b={k:v for k,v in seen[run_id].items() if k!='optimizer_decision'}
                if a!=b:raise ValueError('conflicting shared live outcome')
            if run_id in shared and index>=common_initial_count:
                raise ValueError('adaptive outcomes may not be reused across live optimizers')
            local_ids.add(run_id);seen[run_id]=dict(row)
            environment.add(row.get('environment_hash'));scenario.add(row['scenario_id'])
            c=row.get('cost',{})
            for key in ('accelerator_seconds','wall_seconds'):
                if not isinstance(c.get(key),(int,float)) or not math.isfinite(c[key]) or c[key]<0:
                    raise ValueError('all live failures and successes need actual finite costs')
            cost+=float(c['accelerator_seconds']);wall+=float(c['wall_seconds'])
            cpu+=float(c.get('optimizer_cpu_seconds',0))
            running.append(row);best=measured_incumbent(running)
            curve.append({'method':method,'trial':index+1,'run_id':run_id,
                          'cumulative_accelerator_seconds':cost,
                          'best_feasible_goodput':None if best is None else best['quality_goodput'],
                          'incumbent':best,'optimizer_cpu_seconds':cpu,
                          'method_used':row.get('optimizer_decision',{}).get('method_used',
                                        'common_initial_design' if index<common_initial_count else 'unspecified')})
        curves[method]=curve
        summaries[method]={'trials':len(records),'actual_or_common_debited_accelerator_seconds':cost,
                           'wall_seconds':wall,'optimizer_cpu_seconds':cpu,
                           'adaptive_live_trials':max(0,len(records)-common_initial_count),
                           'best_observed':measured_incumbent(records)}
    if len(environment)!=1 or None in environment:
        raise ValueError('live optimizer comparison requires one declared environment')
    if len(scenario)!=1:raise ValueError('one static workload scenario per optimizer comparison')
    if any(len(x)!=common_initial_count for x in initial_designs):
        raise ValueError('every method must complete the declared common design')
    if any(x!=initial_designs[0] for x in initial_designs[1:]):
        raise ValueError('methods did not receive the same ordered initial configurations')
    horizon=min(s['actual_or_common_debited_accelerator_seconds'] for s in summaries.values())
    checkpoints=[horizon*i/(checkpoint_count-1) for i in range(checkpoint_count)]
    matched=[]
    for point in checkpoints:
        entry={'accelerator_seconds':point,'methods':{}}
        for method,curve in curves.items():
            available=[r for r in curve if r['cumulative_accelerator_seconds']<=point+1e-9]
            entry['methods'][method]={'revealed_trials':len(available),
                'best_feasible_goodput':available[-1]['best_feasible_goodput'] if available else None}
        matched.append(entry)
    costs=[s['actual_or_common_debited_accelerator_seconds'] for s in summaries.values()]
    physical_cost=sum(float(r['cost']['accelerator_seconds']) for r in seen.values())
    return {'source':'measured','mode':'live_optimizer_comparison','methods':summaries,'curves':curves,
            'planned_trace_audit':trace_audit,
            'common_cost_horizon_s':horizon,'cost_matched_checkpoints':matched,
            'final_method_cost_spread_s':max(costs)-min(costs),
            'unique_physical_accelerator_seconds':physical_cost,
            'sum_method_debits_s':sum(costs),'shared_initial_run_ids':sorted(shared),
            'shared_cost_note':('Some common initial runs are physically measured once and equally debited to methods; physical and per-method costs differ.'
                               if shared else 'All optimizer observations are independently measured and charged.'),
            'search_seed_repetitions':1,
            'uncertainty_note':'One live search per method has no between-search-seed confidence interval. Finalist validation remains separate.',
            'environment_hash':next(iter(environment)),'scenario_id':next(iter(scenario))}
