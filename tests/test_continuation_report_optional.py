"""Saved-schema software fixtures only; no optional research experiment is run."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/build_continuation_report.py'
SPEC=importlib.util.spec_from_file_location('optional_continuation_report',SCRIPT)
report=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


@pytest.fixture
def output(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'ROOT',tmp_path)
    monkeypatch.setattr(report,'OUT',tmp_path/'results/continuation')
    return report.OUT


def replay_fixture():
    mode='posthoc_offline_replay_over_measured_table'
    arms=[{'mode':mode,'seed':11,'beta':beta,'initial_config_ids':list(range(6)),
           'initial_source_run_ids':[f'fixture-{i}' for i in range(6)],'adaptive_trials':2,
           'replay_debit_seconds':2400+beta,'best_observed':{'quality_goodput':score}}
          for beta,score in ((1,.3),(0,.25))]
    return {'software_test_fixture':True,'mode':mode,'evidence_source':'measured',
            'is_live_optimizer_experiment':False,'new_hardware_measurements':0,'new_accelerator_seconds_charged':0.,
            'status':'restricted_posthoc_replay_with_fewer_than_four_adaptive_choices',
            'coverage':{'status':'restricted_to_observed_configurations','observed_configurations':8,'expected_configurations':12},
            'seeds':[11,29,47],'initial_points':6,'arms':arms,
            'paired_seed_results':[{'seed':11,'status':'completed','beta1_best_observed_goodput':.3,
                'beta0_best_observed_goodput':.25,'beta1_minus_beta0_goodput':.05,
                'same_initial_configurations':True,'same_initial_source_outcomes':True},
                {'seed':29,'status':'skipped_invalid_grouped_split','reason':'fixture linked-group shortage'}]}


def context_fixture():
    ids=[f'fixture-q{i}' for i in range(54)]
    plan={'software_test_fixture':True,'status':'preregistered_not_run','measurement_question_ids':ids}
    plan['plan_hash']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    arms=[]
    for name,correct,deadline,charge,qag in (('default',27,20,300.01,.2),('tuned',30,25,300.02,.25)):
        arms.append({'arm':name,'run_id':f'fixture-{name}','source':'measured','execution_status':'ok',
                     'measurement_cohort_size':54,'full_measurement_cohort_accuracy':correct/54,
                     'full_measurement_deadline_accuracy':deadline/54,'metrics':{'quality_goodput':qag},
                     'cost':{'accelerator_seconds':charge},'missing_measurement_records':0,
                     'measurement_outcomes':[{'question_id':qid,'correct':i<correct,'deadline_correct':i<deadline}
                                             for i,qid in enumerate(ids)]})
    result={'software_test_fixture':True,'status':'completed','plan_hash':plan['plan_hash'],
            'secondary_only':True,'core_selection_unchanged':True,'official_test_data_accessed':False,
            'arms':arms,'actual_accelerator_seconds':600.03}
    for key,metric in (('paired_accuracy_difference','full_measurement_cohort_accuracy'),
                       ('paired_deadline_accuracy_difference','full_measurement_deadline_accuracy')):
        result[key]={'difference_candidate_minus_reference':arms[1][metric]-arms[0][metric],
                     'confidence':.95,'ci':[-.02,.13],'question_clusters':54,
                     'method':'paired question-cluster percentile bootstrap; equal question weights',
                     'warning':'fixture descriptive interval warning'}
    return plan,result


def test_missing_optional_files_do_not_invent_completed_conditions(output):
    assert 'No saved uncertainty-ablation result' in report.uncertainty_ablation_section()
    assert 'does not establish that either optional arm ran' in report.unseen_context_section()
    assert not output.exists()


def test_replay_reports_paired_observations_partial_coverage_and_skipped_seeds(output):
    save(output/'posthoc-uncertainty-ablation.json',replay_fixture())
    text=report.uncertainty_ablation_section()
    assert 'POSTHOC OFFLINE REPLAY' in text and '8 of 12' in text
    assert '| 11 | completed | 0.300 | 0.250 | 0.050 | 2 / 2 | 2401.00 / 2400.00 |' in text
    assert 'skipped_invalid_grouped_split' in text and 'fixture linked-group shortage' in text
    assert 'no_recorded_replay' in text and 'unavailable' in text
    assert 'never added to the global real budget' in text and 'not new hardware measurements' in text


@pytest.mark.parametrize('fault',['source','live','missing_arm','unpaired','wrong_delta'])
def test_replay_cannot_masquerade_as_live_or_invent_paired_results(output,fault):
    value=replay_fixture()
    if fault=='source':value['evidence_source']='synthetic'
    if fault=='live':value['is_live_optimizer_experiment']=True
    if fault=='missing_arm':value['arms'].pop()
    if fault=='unpaired':value['arms'][1]['initial_source_run_ids'][0]='different'
    if fault=='wrong_delta':value['paired_seed_results'][0]['beta1_minus_beta0_goodput']=.9
    save(output/'posthoc-uncertainty-ablation.json',value)
    with pytest.raises(ValueError):report.uncertainty_ablation_section()

def ceiling_fixture(output):
    """A different-sized artificial table guards against hardcoded campaign claims."""
    source=[{'software_test_fixture':True,'run_id':f'fixture-{i}','config_id':f'config-{i}',
             'scenario_id':'fixture','metrics':{'feasible':True,'quality_goodput':score}}
            for i,score in enumerate((.1,.7,.4))]
    path=output/'search/random/observations.jsonl';path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in source))
    best={'quality_goodput':.7,'config_id':'config-1'}
    arms=[{'seed':seed,'beta':beta,'best_observed':best,
           'curve':[{'best_observed':{'quality_goodput':.1}},{'best_observed':best}],
           'initial_config_ids':['config-0','config-1'],'initial_source_run_ids':['fixture-0','fixture-1']}
          for seed in (4,9) for beta in (0,1)]
    return {'software_test_fixture':True,'arms':arms,'initial_points':2,'seeds':[4,9],
            'table_sha256':hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'coverage':{'observed_configurations':3,'first_observation_provenance':[
                {'selected_run_id':row['run_id'],'config_scenario_key':json.dumps([row['config_id'],'fixture'],separators=(',',':'))}
                for row in source]}}


def test_replay_ceiling_is_derived_from_exact_table_and_pre_adaptation_curves(output):
    value=ceiling_fixture(output)
    text=report.replay_ceiling_note(value)
    assert 'maximum feasible QAG of 0.700' in text and '4 of 4 recorded replay arms' in text
    assert 'after the 2 common starting observations' in text
    assert '`config-1` from `fixture-1` at position 2 (zero-based index 1)' in text
    assert 'All 2 paired replay seeds therefore saturate before adaptive acquisition' in text
    assert 'uninformative about discovery' in text and 'no evidence that uncertainty has no value' in text


def test_equal_replay_endpoints_alone_do_not_establish_initial_saturation(output):
    value=ceiling_fixture(output)
    for arm in value['arms']:arm['curve'][-1]={'best_observed':{'quality_goodput':.4}}
    assert report.replay_ceiling_note(value)==''


def test_replay_ceiling_does_not_guess_missing_source_and_rejects_modified_table(output):
    value=ceiling_fixture(output)
    value['table_sha256']='wrong'
    with pytest.raises(ValueError,match='source-table hash'):report.replay_ceiling_note(value)
    (output/'search/random/observations.jsonl').unlink()
    assert 'no saturation claim is inferred' in report.replay_ceiling_note(value)


def heldout_fixture():
    values={('default',100):.4,('default',200):.6,('tuned',100):.5,('tuned',200):.7}
    arms=[{'arm':name,'allowance':cap,'summary':{'accuracy':score,'deadline_constrained_accuracy':score}}
          for (name,cap),score in values.items()]
    contrasts=[{'endpoint':'deadline_constrained_accuracy','name':name,'estimate':effect,'ci':ci,'confidence':.9}
               for name,effect,ci in (
                   ('tuned_minus_default_short',.1,[-.01,.21]),('tuned_minus_default_long',.1,[-.02,.22]),
                   ('allowance_effect_default',.2,[.05,.35]),('allowance_effect_tuned',.2,[.06,.36]),
                   ('serving_by_allowance_interaction',0.,[-.1,.1]))]
    return arms,contrasts


def test_heldout_interpretation_uses_primary_effects_and_uncertainty_not_fixed_results():
    arms,contrasts=heldout_fixture()
    contrasts.append({'endpoint':'accuracy','name':'tuned_minus_default_short','estimate':999.})
    text=report.heldout_interpretation(arms,contrasts)
    assert '+10.00 percentage points (90% paired interval [-1.00, 21.00]) at 100 tokens' in text
    assert '+20.00 percentage points (90% paired interval [6.00, 36.00]) for tuned' in text
    assert 'Both configuration-effect intervals include zero' in text
    assert 'does not establish that tuning uniquely enables' in text
    assert 'Every correct outcome' in text and 'do not by themselves identify additional completed requests' in text
    assert '5.47' not in text and '999' not in text


def test_heldout_interpretation_does_not_invent_inconclusive_intervals_or_deadline_equality():
    arms,contrasts=heldout_fixture()
    contrasts[0]['ci']=[.01,.21];contrasts[1]['ci']=[.02,.22]
    arms[0]['summary']['accuracy']=.5
    text=report.heldout_interpretation(arms,contrasts)
    assert 'Both configuration-effect intervals include zero' not in text
    assert 'Every correct outcome' not in text


@pytest.mark.parametrize('fault',['missing','duplicate','mismatched_effect','reversed_interval'])
def test_heldout_interpretation_requires_complete_consistent_primary_evidence(fault):
    arms,contrasts=heldout_fixture()
    if fault=='missing':contrasts.pop()
    if fault=='duplicate':contrasts.append(contrasts[0])
    if fault=='mismatched_effect':contrasts[0]['estimate']=.3
    if fault=='reversed_interval':contrasts[0]['ci']=[.2,.1]
    with pytest.raises(ValueError):report.heldout_interpretation(arms,contrasts)


def test_measured_context_reports_actual_arm_metrics_and_saved_paired_intervals(output):
    plan,value=context_fixture()
    save(output/'unseen-context-plan.json',plan);save(output/'unseen-context-summary.json',value)
    text=report.unseen_context_section()
    assert '| default | ok | 54 | 50.00% | 37.04% | 0.200 | 300.010 | 0 |' in text
    assert '| tuned | ok | 54 | 55.56% | 46.30% | 0.250 | 300.020 | 0 |' in text
    assert '| Accuracy | 5.56 | [-2.00, 13.00] | 95.00% | 54 |' in text
    assert 'fixture descriptive interval warning' in text
    assert 'does not access official test data' in text


def test_not_run_partial_and_failed_context_conditions_remain_explicit(output):
    plan,value=context_fixture();save(output/'unseen-context-plan.json',plan)
    partial=copy.deepcopy(value)
    partial.update(status='partially_run',arms=value['arms'][:1],actual_accelerator_seconds=300.01,reason='fixture budget gate')
    partial.pop('paired_accuracy_difference');partial.pop('paired_deadline_accuracy_difference')
    save(output/'unseen-context-summary.json',partial)
    text=report.unseen_context_section()
    assert '**partially_run**' in text and 'fixture budget gate' in text
    assert 'Unavailable' not in text and 'unavailable arm(s): tuned' in text
    assert 'No paired effect is inferred' in text and '| tuned |' not in text
    partial.update(status='not_run',arms=[],actual_accelerator_seconds=0.)
    save(output/'unseen-context-summary.json',partial)
    assert 'No measured optional arm outcome is reported' in report.unseen_context_section()
    failed=copy.deepcopy(value);failed['status']='completed_with_failed_runs'
    failed['arms'][1]['execution_status']='failed'
    save(output/'unseen-context-summary.json',failed)
    text=report.unseen_context_section()
    assert '**completed_with_failed_runs**' in text and '| tuned | failed |' in text
    assert 'Failed attempts remain labeled' in text


@pytest.mark.parametrize('fault',['source','plan','accuracy','cost','paired_difference','missing_arm','cohort','seeds'])
def test_context_provenance_complete_denominator_and_pairing_are_checked(output,fault):
    plan,value=context_fixture()
    if fault=='source':value['arms'][0]['source']='synthetic'
    if fault=='plan':plan['status']='changed-after-hash'
    if fault=='accuracy':value['arms'][0]['full_measurement_cohort_accuracy']=.99
    if fault=='cost':value['actual_accelerator_seconds']=0.
    if fault=='paired_difference':value['paired_accuracy_difference']['difference_candidate_minus_reference']=.99
    if fault=='missing_arm':value['arms'].pop();value['actual_accelerator_seconds']=300.01
    if fault=='cohort':value['arms'][1]['measurement_outcomes'][0]['question_id']='not-in-frozen-cohort'
    if fault=='seeds':value['arms'][1]['measurement_outcomes'][0]['sampling_seed']=999
    save(output/'unseen-context-plan.json',plan);save(output/'unseen-context-summary.json',value)
    with pytest.raises(ValueError):report.unseen_context_section()
