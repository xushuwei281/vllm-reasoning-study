import copy
import pytest
from vllm_tuner.live_analysis import summarize_live_search, validate_planned_trace_hashes


def row(method,index,cost=10,goodput=.1,feasible=True):
    return {'run_id':f'{method}-{index}','config_id':f'c{index}','scenario_id':'s','source':'measured',
            'partition':'train','environment_hash':'e','workload':{},
            'metrics':{'quality_goodput':goodput,'feasible':feasible},
            'cost':{'accelerator_seconds':cost,'wall_seconds':cost,'optimizer_cpu_seconds':.01}}


def test_common_cost_uses_only_available_observations_and_failed_costs():
    histories={'random':[row('r',0,10,.1),row('r',1,10,0,False),row('r',2,10,.3)],
               'neural':[row('n',0,10,.1),row('n',1,15,.2)]}
    report=summarize_live_search(histories,common_initial_count=1,checkpoint_count=6)
    assert report['common_cost_horizon_s']==25
    last=report['cost_matched_checkpoints'][-1]['methods']
    assert last['random']=={'revealed_trials':2,'best_feasible_goodput':.1}
    assert last['neural']['best_feasible_goodput']==.2
    assert report['unique_physical_accelerator_seconds']==55
    assert report['final_method_cost_spread_s']==5


def test_shared_initial_evidence_has_one_physical_cost_not_double():
    initial=row('shared',0)
    histories={'random':[initial,row('r',1)],'gp':[copy.deepcopy(initial),row('g',1)]}
    with pytest.raises(ValueError):summarize_live_search(histories,common_initial_count=1)
    report=summarize_live_search(histories,common_initial_count=1,permitted_shared_run_ids=['shared-0'])
    assert report['unique_physical_accelerator_seconds']==30
    assert report['sum_method_debits_s']==40


def test_no_shared_adaptive_outcomes_or_test_outcomes():
    r=row('shared',1)
    histories={'random':[row('r',0),r],'gp':[row('g',0),copy.deepcopy(r)]}
    with pytest.raises(ValueError):summarize_live_search(histories,common_initial_count=1,permitted_shared_run_ids=[r['run_id']])
    histories={'random':[row('r',0)],'gp':[row('g',0)]}
    histories['gp'][0]['partition']='holdout'
    with pytest.raises(ValueError):summarize_live_search(histories,common_initial_count=1)


def test_incomparable_starts_and_hardware_rejected():
    histories={'random':[row('r',0)],'gp':[row('g',1)]}
    with pytest.raises(ValueError):summarize_live_search(histories,common_initial_count=1)
    histories={'random':[row('r',0)],'gp':[row('g',0)]}
    histories['gp'][0]['environment_hash']='different'
    with pytest.raises(ValueError):summarize_live_search(histories,common_initial_count=1)


def test_planned_trace_audit_retains_charged_failure_without_measured_hash():
    histories = {method: [row(method, index) for index in range(3)] for method in ('random', 'gp', 'neural')}
    expected = ['schedule-a', 'schedule-b', 'schedule-c']
    for records in histories.values():
        for index, record in enumerate(records):
            record.update(status='ok', trace_hash=expected[index], trace_id=expected[index])
    failure = histories['gp'][1]
    failure.update(status='failed', failure='startup timed out')
    failure.pop('trace_hash'); failure.pop('trace_id')
    failure['metrics'].update(quality_goodput=0, feasible=False)
    report = summarize_live_search(histories, common_initial_count=2, expected_trace_hashes=expected)
    audit = report['planned_trace_audit']
    assert audit['methods']['gp']['matched_trials'] == 2
    assert audit['methods']['gp']['unverified_failed_trials'] == 1
    missing = audit['methods']['gp']['trials'][1]
    assert missing['actual_trace_hash'] is None and missing['expected_trace_hash'] == 'schedule-b'
    assert missing['audit_status'] == 'unverified_failure'
    assert report['unique_physical_accelerator_seconds'] == 90
    assert report['methods']['gp']['actual_or_common_debited_accelerator_seconds'] == 30
    assert len(report['curves']['gp']) == 3
    assert report['curves']['gp'][1]['cumulative_accelerator_seconds'] == 20
    assert summarize_live_search(histories, common_initial_count=2)['planned_trace_audit']['enabled'] is False


def test_planned_trace_mismatch_missing_success_and_linking_identity_rejected():
    records = [row('r', index) for index in range(2)]
    for index, record in enumerate(records):
        record.update(status='ok', trace_hash=f'schedule-{index}', trace_id=f'schedule-{index}')
    expected = ['schedule-0', 'schedule-1']
    changed = copy.deepcopy(records); changed[1]['trace_hash'] = 'different'
    with pytest.raises(ValueError, match='differs from the frozen plan'):
        summarize_live_search({'random': changed}, common_initial_count=1, expected_trace_hashes=expected)
    changed[1]['status'] = 'failed'; changed[1]['metrics']['feasible'] = False
    with pytest.raises(ValueError, match='differs from the frozen plan'):
        summarize_live_search({'random': changed}, common_initial_count=1, expected_trace_hashes=expected)
    changed = copy.deepcopy(records); changed[1].pop('trace_hash')
    with pytest.raises(ValueError, match='lacks an actual trace hash'):
        summarize_live_search({'random': changed}, common_initial_count=1, expected_trace_hashes=expected)
    changed = copy.deepcopy(records); changed[1]['trace_id'] = changed[0]['trace_id']
    with pytest.raises(ValueError, match='repeated trace_id'):
        summarize_live_search({'random': changed}, common_initial_count=1, expected_trace_hashes=expected)
    with pytest.raises(ValueError, match='every observed trial index'):
        summarize_live_search({'random': records}, common_initial_count=1, expected_trace_hashes=expected[:1])


def test_planned_trace_precheck_can_run_before_any_gpu_trial():
    assert validate_planned_trace_hashes(['a', 'b'], required_count=2) == ('a', 'b')
    for invalid in ([], ['same', 'same'], ['', 'b'], ['a', None], 'not-a-sequence-of-hashes'):
        with pytest.raises(ValueError, match='distinct nonempty|distinct within'):
            validate_planned_trace_hashes(invalid)
