"""Synthetic software tests of report refusal guards; no research results."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/build_continuation_report.py'
SPEC=importlib.util.spec_from_file_location('continuation_report',SCRIPT)
report=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(report.canonical(value))


@pytest.fixture
def neural_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    out=tmp_path/'results/continuation';monkeypatch.setattr(report,'OUT',out)
    history=[]
    for i in range(10):
        key=json.dumps([f'config{i}','software-test'],separators=(',',':'))
        row={'run_id':f'live-neural-{i:02d}','config_id':f'config{i}','scenario_id':'software-test',
             'config':{'max_num_batched_tokens':512,'max_num_seqs':8+i},'workload':{'profile':'software-test'},
             'partition':'train','source':'measured','software_test_fixture':True,
             'metrics':{'quality_goodput':.1+i*.01,'feasible':True}}
        if i<6:
            row['optimizer_decision']={'method_used':'shared_initial_design'}
        else:
            ordinal=i-5
            prediction={'goodput_mean':.2,'goodput_std':.01,'feasible_probability':.8}
            training={'source':'measured','training_records':i-1,'validation_records':1,
                      'method_version':'mlp-ensemble-v2-train-only-log-target-standardization',
                      'members':[{'best_epoch':1,'epochs':[{'epoch':1,'train_loss':1.,'validation_loss':1.2}]} for _ in range(3)]}
            provenance={'ordinal':ordinal,'source':'measured','training_report':training,
                'selected_key':key,'selected_features':{k:row[k] for k in ('config_id','scenario_id','config','workload')},
                'selected_prediction':prediction,'acquisition':.168,
                'observed_sha256':hashlib.sha256(report.canonical(history)).hexdigest()}
            signature=hashlib.sha256(report.canonical(provenance)).hexdigest()
            name=f'proposal-{ordinal:03d}-{signature[:12]}'
            destination=out/'search/neural/models'/name
            write(destination/'observations.json',history)
            write(destination/'report.json',{'provenance':provenance,'diagnostics':{}})
            (destination/'ensemble.pt').write_bytes(b'SOFTWARE-TEST CHECKSUM FIXTURE; NOT A MODEL')
            write(destination/'manifest.json',{'proposal_sha256':signature,'files':{
                p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}})
            remote=Path('/workspace/study/results/continuation/search/neural/models')/name
            row['optimizer_decision']={'method_used':'neural','requested_method':'neural',
                'observations_seen':i,'selected_key':key,'prediction':prediction,'acquisition':.168,
                'model_artifact':{'directory':str(remote),'report':str(remote/'report.json'),
                    'checkpoint':str(remote/'ensemble.pt'),'proposal_sha256':signature}}
        history.append(copy.deepcopy(row))
    path=out/'search/neural/observations.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in history))
    return out,history


def test_four_actual_guided_rows_and_corresponding_artifacts_required(neural_fixture):
    out,history=neural_fixture
    result,reports=report.neural_audit(make_plot=False)
    assert result['verified_guided_proposal_count']==4
    assert result['verified_artifact_report_count']==4
    assert [row['observations_seen'] for row in result['prospective_selected_proposals']]==[6,7,8,9]
    assert len(reports)==4


def test_random_fallback_cannot_be_reported_as_neural_guidance(neural_fixture):
    out,history=neural_fixture
    history[6]['optimizer_decision']['method_used']='random_neural_warm_start'
    (out/'search/neural/observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    with pytest.raises(RuntimeError,match='genuinely neural-guided'):
        report.neural_audit(make_plot=False)
    assert not (out/'analysis/neural-prospective-audit.json').exists()


def test_four_decisions_with_only_three_reports_are_refused(neural_fixture):
    out,_=neural_fixture
    next((out/'search/neural/models').glob('proposal-004-*/report.json')).unlink()
    with pytest.raises(RuntimeError,match='Exactly four'):
        report.neural_audit(make_plot=False)


def test_missing_checkpoint_in_manifest_is_refused(neural_fixture):
    out,_=neural_fixture
    path=next((out/'search/neural/models').glob('proposal-001-*/manifest.json'))
    manifest=report.read(path);manifest['files'].pop('ensemble.pt');write(path,manifest)
    with pytest.raises(ValueError,match='complete checkpoint'):
        report.neural_audit(make_plot=False)


def test_decision_prediction_must_match_retained_prospective_artifact(neural_fixture):
    out,history=neural_fixture
    history[6]['optimizer_decision']['prediction']['goodput_mean']=99.
    (out/'search/neural/observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    with pytest.raises(ValueError,match='prospective decision'):
        report.neural_audit(make_plot=False)


def test_artifact_observations_cannot_differ_from_history_prefix(neural_fixture):
    out,history=neural_fixture
    history[0]['metrics']['quality_goodput']=99.
    (out/'search/neural/observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    with pytest.raises(ValueError,match='prior observations'):
        report.neural_audit(make_plot=False)


@pytest.fixture
def completed_summary_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path);monkeypatch.setattr(report,'OUT',tmp_path/'results/continuation')
    names=['results/continuation/search-plan.json','results/measured/budget.jsonl',
        'results/continuation/analysis-spec.json','results/continuation/freeze.json',
        'results/continuation/validation-status.json','results/continuation/independent-validation-evidence.json',
        'results/continuation/cloud-accounting-final.json']
    names += [f'results/continuation/search/{method}/observations.jsonl' for method in ('random','gp','neural')]
    for name in names:write(tmp_path/name,{'software_test_fixture':True})
    analyzer=tmp_path/'scripts/analyze_continuation.py';analyzer.parent.mkdir();analyzer.write_text('# software test')
    figures=['live_goodput_vs_allocation.png','live_goodput_vs_trials.png','configuration_goodput_ranges.png','latency_goodput.png','heldout_accuracy_and_tokens.png']
    for name in figures:
        path=tmp_path/'results/continuation/analysis'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'SOFTWARE TEST, NOT A FIGURE')
    summary={'source':'measured','software_test_fixture':True,'report_status':'complete','completion_blockers':[],
        'accounting':{'active_run_ids':[],'global_held_accelerator_seconds':0,'terminal_cloud':{'verified':True,'record':{'estimated_charge_usd':6.}},
            'each_global_finish_counted_once':True,'original_budget_accelerator_seconds':28800,'global_spent_accelerator_seconds':15000.},
        'heldout':{'four_arms_per_seed_available':True,'arms':[{'arm':arm,'allowance':allowance,
            'sampling_seed_base':724,'run_id':f'{arm}-{allowance}','summary':{'cohort_size':128}}
            for arm in ('default','tuned') for allowance in (2048,4096)]},
        'search':{'planned_trials_complete':True,'allocation_audit':{'complete':True},
                  'methods':{method:{'trials':10} for method in ('random','gp','neural')}},
        'input_sha256':{name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in names},
        'analyzer_sha256':hashlib.sha256(analyzer.read_bytes()).hexdigest(),'plot_files':figures}
    return tmp_path,summary


def test_completed_summary_must_match_current_input_bytes(completed_summary_fixture):
    root,summary=completed_summary_fixture
    report.verify_report_evidence(summary)
    (root/'results/measured/budget.jsonl').write_text('changed')
    with pytest.raises(RuntimeError,match='input changed'):
        report.verify_report_evidence(summary)


@pytest.mark.parametrize('fault',['terminal','arms','plots','charge','source','trials'])
def test_complete_label_alone_cannot_bypass_guards(completed_summary_fixture,fault):
    _,summary=completed_summary_fixture
    if fault=='terminal':summary['accounting']['terminal_cloud']['verified']=False
    if fault=='arms':summary['heldout']['arms'].pop()
    if fault=='plots':summary['plot_files']=[]
    if fault=='charge':summary['accounting']['terminal_cloud']['record']['estimated_charge_usd']=10.
    if fault=='source':summary['source']='synthetic'
    if fault=='trials':summary['search']['methods']['neural']['trials']=9
    with pytest.raises(RuntimeError):report.verify_report_evidence(summary)


def test_missing_optional_stability_is_not_invented(completed_summary_fixture):
    root,_=completed_summary_fixture
    assert report.stability_section()==''
    path=root/'results/continuation/analysis/stability-summary.json'
    write(path,{'source':'measured','phase':'independent_development_validation','status':'not_available'})
    assert 'no completed stability result is inferred' in report.stability_section()


def test_optional_stability_complete_label_requires_plot_and_both_arms(completed_summary_fixture):
    root,_=completed_summary_fixture
    write(root/'results/continuation/analysis/stability-summary.json',
          {'source':'measured','phase':'independent_development_validation','status':'complete','arms':[]})
    with pytest.raises(ValueError,match='both arm records'):
        report.stability_section()


def provenance_fixture(adaptive_value):
    """Tiny labelled software fixture for discovery-phase attribution only."""
    history=[]
    for index in range(10):
        value=.4 if index==5 else adaptive_value if index==7 else .2
        history.append({'source':'synthetic','software_test_fixture':True,
            'run_id':f'live-neural-{index:02d}','config_id':f'config{index}',
            'metrics':{'feasible':True,'quality_goodput':value},
            'optimizer_decision':{'method_used':'shared_initial_design' if index<6 else 'neural'}})
    index=7 if adaptive_value>.4 else 5
    search={'methods':{'neural':{'incumbent_run_id':history[index]['run_id'],
                               'best_feasible_quality_goodput':max(.4,adaptive_value)}}}
    return history,search


def test_neural_history_winner_from_common_starts_is_not_guided_discovery():
    history,search=provenance_fixture(.3)
    result=report.incumbent_origins({'neural':history},search,history[5])
    selected=result['selected_finalist']
    assert selected['phase']=='shared_initial_design' and selected['trial']==6
    assert result['methods']['neural']['incremental_best_goodput_after_initial']==0.
    assert 'not evidence of neural-guided discovery' in report.discovery_note(selected)


def test_adaptive_discovery_is_identified_from_actual_decision():
    history,search=provenance_fixture(.45)
    result=report.incumbent_origins({'neural':history},search,history[7])
    selected=result['selected_finalist']
    assert selected['phase']=='adaptive' and selected['trial']==8
    assert selected['decision_method_used']=='neural'
    assert result['methods']['neural']['incremental_best_goodput_after_initial']==pytest.approx(.05)
    assert 'adaptive trial 8' in report.discovery_note(selected)


def test_later_adaptive_tie_does_not_erase_initial_origin_or_create_gain():
    history,search=provenance_fixture(.4)
    result=report.incumbent_origins({'neural':history},search,history[5])
    method=result['methods']['neural']
    assert method['incumbent']['phase']=='shared_initial_design'
    assert len(method['all_tied_maximum_observations'])==2
    assert method['incremental_best_goodput_after_initial']==0.
    search['methods']['neural']['incumbent_run_id']=history[7]['run_id']
    with pytest.raises(ValueError,match='first observation'):
        report.incumbent_origins({'neural':history},search,history[5])


def test_optimizer_resource_text_separates_clocks_and_retains_tiny_cap_excess():
    search={'methods':{'neural':{'optimizer_cpu_seconds':2.,'optimizer_wall_seconds':3.,'missing_optimizer_wall_time_count':0}},
        'optimizer_timing':{'optimizer_cpu_seconds_sum':2.,'optimizer_wall_seconds_sum':3.},
        'method_cap_audit':{'declared_base_trial_admission_cap_seconds':3175.,'methods':{'neural':{
            'base_trial_accelerator_seconds':3000.,'fully_attributed_accelerator_seconds':3175.000001,
            'declared_base_trial_admission_cap_seconds':3175.,'fully_attributed_excess_seconds':.000001,
            'status':'exceeds_declared_amount'}}}}
    text=report.optimizer_resource_section(search)
    assert '2.000' in text and '3.000' in text and '1e-06' in text
    assert 'SearchPolicy.choose' in text
    assert 'neither is added again to GPU allocation' in text
    assert 'does not claim that the code enforced' in text
    assert 'global ledger can show zero optimizer CPU annotations' in text
    search['methods']['neural']['optimizer_wall_seconds']=None
    search['methods']['neural']['missing_optimizer_wall_time_count']=1
    search['optimizer_timing']['optimizer_wall_seconds_sum']=None
    assert 'unavailable' in report.optimizer_resource_section(search)


def test_terminal_prose_distinguishes_charged_bound_from_unknown_actual_end():
    terminal={'duration_basis':'terminal_observation_upper_bound','charge_upper_bound':True,
        'terminal_status':{'stage':'CANCELED'},'provider_finished_at':None,'finished_at':None,
        'actual_accelerator_seconds':None,'accounting_end_at':'2026-01-01T00:00:38+00:00',
        'charged_accelerator_seconds':38.}
    words=report.terminal_accounting_prose(terminal,78.)
    assert words['upper_bound'] is True
    assert words['total_label']=='TOTAL CHARGED (upper bound)'
    assert 'conservative upper bound' in words['summary']
    assert 'confirmed canceled' in words['summary']
    assert 'actual finish timestamp and exact allocation duration remain unknown' in words['summary']
    assert 'not represented as the actual job end' in words['detail']
    assert 'Actual allocation could be shorter' in words['detail']
    assert 'exact actual duration remain unknown' in words['reproduction']
    terminal['actual_accelerator_seconds']=38.
    with pytest.raises(RuntimeError,match='unknown actual finish'):
        report.terminal_accounting_prose(terminal,78.)


def test_legacy_provider_finish_accounting_prose_remains_supported():
    words=report.terminal_accounting_prose({'finished_at':'2026-01-01T00:00:38+00:00'},78.)
    assert words['upper_bound'] is False
    assert words['total_label']=='TOTAL'
    assert 'provider creation and finish timestamps' in words['detail']
    assert 'unknown' not in words['summary']
