"""Synthetic SOFTWARE TEST fixtures only; no measured or final results created."""
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import pytest

MODULE_PATH=Path(__file__).resolve().parents[1]/'scripts'/'analyze_continuation.py'
SPEC=importlib.util.spec_from_file_location('continuation_analysis',MODULE_PATH)
analysis=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value, lines=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in value) if lines else json.dumps(value))


def debit(run, seconds):
    return [{'event':'begin','run_id':run,'devices':1,'max_wall_seconds':seconds,'phase':'search'},
            {'event':'finish','run_id':run,'wall_seconds':seconds,'accelerator_seconds':seconds}]


@pytest.fixture
def software_fixture(tmp_path):
    out=tmp_path/'results'/'continuation'
    spec={'heldout_results_seen':False,'bootstrap':{'samples':2000,'seed':724,'confidence':.95},
          'contrasts':[{'name':'allowance_effect_default','weights':{'default_long':1,'default_short':-1}},
                       {'name':'interaction','weights':{'tuned_long':1,'tuned_short':-1,'default_long':-1,'default_short':1}}]}
    spec['spec_sha256']=analysis.digest(spec)
    write(out/'analysis-spec.json',spec)
    write(out/'search-plan.json',{'trials_per_method':1,'method_budget_s':3175.})
    events=[{'event':'policy','total_accelerator_seconds':1000.,'validation_fraction':.25}]
    events+=debit('pilot-metal-software-test',40.)
    events+=debit('cloud-overhead-before-main',2.)
    for index,method in enumerate(analysis.METHODS):
        run=f'live-{method}-00'
        row={'source':'synthetic','partition':'train','run_id':run,'config_id':'same-config',
             'config':{'max_num_seqs':8,'max_num_batched_tokens':512},'environment_hash':'software-test',
             'status':'ok','cost':{'accelerator_seconds':10.,'wall_seconds':10.,'optimizer_cpu_seconds':1.,'optimizer_wall_seconds':1.5},
             'measurement_duration_s':5.,'metrics':{'quality_goodput':.2+index*.1,'feasible':True,
                 'quality_slo_completed_in_window':1,'p99_e2e_s':2.,'tail_sample_warning':True}}
        write(out/'search'/method/'observations.jsonl',[row],lines=True)
        events+=debit(run,10.)+debit(f'cloud-overhead-round00-{method}',2.)
    write(tmp_path/'results'/'measured'/'budget.jsonl',events,lines=True)
    created=datetime(2026,1,1,tzinfo=timezone.utc)
    write(out/'cloud-allocation.json',{'id':'software-test-job','created_at':created.isoformat(),
        'devices':1,'preallocation_charged_accelerator_seconds':40.})
    write(out/'cloud-accounting-final.json',{'allocation_id':'software-test-job','created_at':created.isoformat(),
        'finished_at':(created+timedelta(seconds=38)).isoformat(),'reconciled':True,
        'actual_accelerator_seconds':38.,'global_ledger_spent_accelerator_seconds':78.})
    return tmp_path,events,spec


def test_tagged_overheads_added_to_derived_curves_exactly_once(software_fixture):
    root,events,spec=software_fixture
    path=root/'results'/'continuation'/'search'/'random'/'observations.jsonl'
    before=path.read_bytes()
    summary=analysis.summarize(root,software_test=True)
    assert summary['source']=='software_test'
    assert path.read_bytes()==before
    assert summary['accounting']['global_spent_accelerator_seconds']==78.
    assert sum(summary['accounting']['category_accelerator_seconds'].values())==78.
    assert summary['accounting']['category_accelerator_seconds']['shared_setup_before_main']==2.
    assert summary['search']['allocation_audit']['assigned_overhead_event_count']==3
    assert summary['search']['allocation_audit']['assigned_overhead_accelerator_seconds']==6.
    for method in analysis.METHODS:
        row=summary['search']['curves'][method][0]
        assert row['cumulative_accelerator_seconds']==12.
        assert row['raw_trial_accelerator_seconds']==10.
        assert row['measurement_duration_seconds']==5.
        assert row['window_correct_per_whole_trial_accelerator_second']==pytest.approx(1/12)
        assert summary['search']['methods'][method]['optimizer_wall_seconds']==1.5
    assert summary['search']['optimizer_timing']['optimizer_cpu_seconds_sum']==3.
    assert summary['search']['optimizer_timing']['optimizer_wall_seconds_sum']==4.5
    repeated=summary['search']['configuration_repeats'][0]
    assert repeated['count']==3
    assert repeated['quality_goodput_mean']==pytest.approx(.3)
    assert repeated['quality_goodput_min']==pytest.approx(.2)
    assert repeated['quality_goodput_max']==pytest.approx(.4)
    assert 'not confidence' in repeated['interval_kind']


def test_real_analysis_refuses_synthetic_software_fixture(software_fixture):
    root,*_=software_fixture
    with pytest.raises(ValueError,match='live evidence'):
        analysis.summarize(root)


def test_terminal_reconciliation_alone_cannot_claim_final_complete(software_fixture):
    root,*_=software_fixture
    summary=analysis.summarize(root,software_test=True)
    assert summary['accounting']['terminal_cloud']['verified'] is True
    assert summary['report_status']=='incomplete'
    assert summary['heldout']['four_arms_per_seed_available'] is False
    assert any('four' in reason for reason in summary['completion_blockers'])


def test_missing_and_orphan_overheads_are_visible_not_reassigned(software_fixture):
    root,events,_=software_fixture
    events=[e for e in events if e.get('run_id')!='cloud-overhead-round00-random']
    events+=debit('cloud-overhead-round99-random',2.)
    write(root/'results'/'measured'/'budget.jsonl',events,lines=True)
    summary=analysis.summarize(root,software_test=True)
    audit=summary['search']['allocation_audit']
    assert not audit['complete']
    assert audit['missing_overhead_run_ids']==['cloud-overhead-round00-random']
    assert audit['unmatched_tagged_overhead_run_ids']==['cloud-overhead-round99-random']
    assert summary['search']['methods']['random']['total_attributed_accelerator_seconds']==10.
    assert summary['accounting']['global_spent_accelerator_seconds']==78.


def test_duplicate_global_finish_or_trial_cost_mismatch_rejected(software_fixture):
    root,events,_=software_fixture
    with pytest.raises(ValueError,match='duplicate finish'):
        analysis.audit_ledger(events+[copy.deepcopy(events[-1])])
    path=root/'results'/'continuation'/'search'/'gp'/'observations.jsonl'
    row=json.loads(path.read_text());row['cost']['accelerator_seconds']=9.
    write(path,[row],lines=True)
    with pytest.raises(ValueError,match='ledger cost differ'):
        analysis.summarize(root,software_test=True)


def test_terminal_duration_and_active_reservations_checked(software_fixture):
    root,events,_=software_fixture
    path=root/'results'/'continuation'/'cloud-accounting-final.json'
    terminal=json.loads(path.read_text());terminal['actual_accelerator_seconds']=37.
    write(path,terminal)
    events.append({'event':'begin','run_id':'unfinished-software-test','devices':1,'max_wall_seconds':5.})
    write(root/'results'/'measured'/'budget.jsonl',events,lines=True)
    summary=analysis.summarize(root,software_test=True)
    audit=summary['accounting']['terminal_cloud']
    assert not audit['verified']
    assert any('duration' in item for item in audit['reasons'])
    assert any('active' in item for item in audit['reasons'])
    assert summary['accounting']['global_held_accelerator_seconds']==5.


def upper_bound_record(path):
    terminal=json.loads(path.read_text())
    endpoint=terminal['finished_at']
    terminal.update(duration_basis='terminal_observation_upper_bound',charge_upper_bound=True,
        provider_finished_at=None,finished_at=None,actual_accelerator_seconds=None,
        charged_accelerator_seconds=38.,accounting_end_at=endpoint,terminal_observed_at=endpoint,
        terminal_status={'stage':'CANCELED'})
    return terminal


@pytest.mark.parametrize('stage',['CANCELED','ERROR','COMPLETED'])
def test_confirmed_terminal_observation_supports_conservative_charge(software_fixture,stage):
    root,_,_=software_fixture
    path=root/'results/continuation/cloud-accounting-final.json'
    terminal=upper_bound_record(path);terminal['terminal_status']['stage']=stage;write(path,terminal)
    before=path.read_bytes()
    summary=analysis.summarize(root,software_test=True)
    audit=summary['accounting']['terminal_cloud']
    assert audit['verified'] is True
    assert audit['charge_upper_bound'] is True
    assert audit['charged_accelerator_seconds']==38.
    assert audit['actual_accelerator_seconds'] is None
    assert audit['record']['finished_at'] is None
    assert 'actual allocation duration is unknown' in audit['accounting_note']
    assert summary['accounting']['global_spent_accelerator_seconds']==78.
    assert summary['report_status']=='incomplete'  # No held-out evidence in software fixture.
    assert path.read_bytes()==before


@pytest.mark.parametrize('fault',['running','missing_upper_flag','invented_finish','invented_provider_finish',
    'invented_actual','missing_actual','different_observation','naive_endpoint','non_utc_endpoint',
    'wrong_charge','wrong_total','nonterminal_status'])
def test_upper_bound_requires_terminal_evidence_and_preserves_unknown_duration(software_fixture,fault):
    root,_,_=software_fixture
    path=root/'results/continuation/cloud-accounting-final.json'
    terminal=upper_bound_record(path)
    if fault=='running':terminal['terminal_status']['stage']='RUNNING'
    elif fault=='nonterminal_status':terminal['terminal_status']=None
    elif fault=='missing_upper_flag':terminal.pop('charge_upper_bound')
    elif fault=='invented_finish':terminal['finished_at']=terminal['accounting_end_at']
    elif fault=='invented_provider_finish':terminal['provider_finished_at']=terminal['accounting_end_at']
    elif fault=='invented_actual':terminal['actual_accelerator_seconds']=38.
    elif fault=='missing_actual':terminal.pop('actual_accelerator_seconds')
    elif fault=='different_observation':terminal['terminal_observed_at']='2026-01-01T00:00:39+00:00'
    elif fault=='naive_endpoint':terminal['accounting_end_at']='2026-01-01T00:00:38'
    elif fault=='non_utc_endpoint':terminal['accounting_end_at']='2026-01-01T08:00:38+08:00'
    elif fault=='wrong_charge':terminal['charged_accelerator_seconds']=37.
    elif fault=='wrong_total':terminal['global_ledger_spent_accelerator_seconds']=79.
    write(path,terminal)
    audit=analysis.summarize(root,software_test=True)['accounting']['terminal_cloud']
    assert audit['verified'] is False
    assert audit['reasons']


def test_missing_policy_wall_timer_stays_unknown(software_fixture):
    root,_,_=software_fixture
    path=root/'results/continuation/search/gp/observations.jsonl'
    row=json.loads(path.read_text());row['cost'].pop('optimizer_wall_seconds')
    write(path,[row],lines=True)
    summary=analysis.summarize(root,software_test=True)
    method=summary['search']['methods']['gp']
    assert method['optimizer_wall_seconds'] is None
    assert method['missing_optimizer_wall_time_count']==1
    assert summary['search']['optimizer_timing']['optimizer_wall_seconds_sum'] is None
    assert summary['accounting']['global_spent_accelerator_seconds']==78.


def test_3175_cap_audit_separates_base_from_attributed_without_rewriting_cost(software_fixture):
    root,_,_=software_fixture
    histories={method:analysis.read_jsonl(root/f'results/continuation/search/{method}/observations.jsonl') for method in analysis.METHODS}
    events=[{'event':'policy','total_accelerator_seconds':28800.,'validation_fraction':.25}]
    for method,items in histories.items():
        items[0]['cost'].update(accelerator_seconds=3174.,wall_seconds=3174.,optimizer_wall_seconds=.5)
        events+=debit(items[0]['run_id'],3174.)+debit(f'cloud-overhead-round00-{method}',1.000001)
    before=copy.deepcopy(histories)
    result,_,_=analysis.search_summary(histories,analysis.audit_ledger(events),
        {'trials_per_method':1,'method_budget_s':3175.},software_test=True)
    audit=result['method_cap_audit']
    assert histories==before
    assert audit['affects_completion_gate'] is False
    assert result['planned_trials_complete'] is True and result['allocation_audit']['complete'] is True
    for value in audit['methods'].values():
        assert value['base_trial_within_declared_amount'] is True
        assert value['fully_attributed_within_declared_amount'] is False
        assert value['fully_attributed_excess_seconds']==pytest.approx(.000001)
        assert value['status']=='exceeds_declared_amount'


def test_descriptive_method_cap_flag_does_not_add_completion_blocker(software_fixture):
    root,_,_=software_fixture
    before=analysis.summarize(root,software_test=True)
    write(root/'results/continuation/search-plan.json',{'trials_per_method':1,'method_budget_s':11.999999})
    after=analysis.summarize(root,software_test=True)
    assert after['completion_blockers']==before['completion_blockers']
    assert after['report_status']==before['report_status']
    assert all(row['status']=='exceeds_declared_amount' for row in after['search']['method_cap_audit']['methods'].values())


def test_preregistered_paired_bootstrap_is_reproducible_and_pair_strict(software_fixture):
    _,_,spec=software_fixture
    # Tiny synthetic software-test outcomes remain in memory; never final artifacts.
    arms=[]
    for arm,allowance,correct in [('default',2,[False,False]),('default',4,[True,False]),
                                 ('tuned',2,[False,False]),('tuned',4,[True,True])]:
        arms.append({'arm':arm,'allowance':allowance,'summary':{'question_outcomes':[
            {'question_id':f'software-test-q{i}','sampling_seed':724+i,'correct':value,'deadline_correct':value}
            for i,value in enumerate(correct)]}})
    first=analysis.paired_contrasts(arms,spec)
    assert first==analysis.paired_contrasts(arms,spec)
    assert all(row['estimate']==.5 and row['question_clusters']==2 for row in first)
    arms[-1]['summary']['question_outcomes'][0]['sampling_seed']=999
    with pytest.raises(ValueError,match='not identical'):
        analysis.paired_contrasts(arms,spec)
