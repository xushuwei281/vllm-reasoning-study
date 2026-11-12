"""Generate the completed research report from audited continuation artifacts."""
from pathlib import Path
import json, math, hashlib, re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/continuation'

def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def f(value,n=3):return 'unavailable' if value is None else f'{value:.{n}f}'
def pct(value):return 'unavailable' if value is None else f'{100*value:.2f}%'
def table(headers,values):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in values])
def config(row):
    c=row['config'];return f"batch {c.get('max_num_batched_tokens') or 'default'}, seq {c.get('max_num_seqs') or 'default'}"

def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def verify_report_evidence(summary):
    """Fail before writing a completed report if evidence is stale or incomplete."""
    if summary.get('source')!='measured' or summary.get('report_status')!='complete' or summary.get('completion_blockers'):
        raise RuntimeError('Refusing a completed report before all measured evidence and terminal accounting exist')
    acc=summary['accounting'];heldout=summary['heldout'];search=summary['search']
    if (acc.get('active_run_ids') or acc.get('global_held_accelerator_seconds',0)!=0
            or acc.get('terminal_cloud',{}).get('verified') is not True
            or acc.get('each_global_finish_counted_once') is not True
            or acc.get('original_budget_accelerator_seconds')!=28800
            or not 0<=acc['global_spent_accelerator_seconds']<=28800):
        raise RuntimeError('Completed report requires reconciled terminal allocation within the original budget')
    terminal=acc['terminal_cloud']['record']
    charge=terminal.get('estimated_charge_usd',terminal.get('estimated_total_charge_usd'))
    if charge is not None and (isinstance(charge,bool) or not isinstance(charge,(int,float))
                              or not math.isfinite(charge) or not 0<=charge<10):
        raise RuntimeError('Estimated cloud charge does not meet the authorized limit')
    arms=heldout.get('arms',[])
    required={(name,tokens) for name in ('default','tuned') for tokens in (2048,4096)}
    if (heldout.get('four_arms_per_seed_available') is not True or len(arms)!=4
            or {(row['arm'],row['allowance']) for row in arms}!=required
            or len({row['run_id'] for row in arms})!=4
            or any(row.get('sampling_seed_base')!=724 or row['summary'].get('cohort_size')!=128 for row in arms)):
        raise RuntimeError('Report template requires exactly four distinct frozen 128-question arms at seed 724')
    if (search.get('planned_trials_complete') is not True or search['allocation_audit'].get('complete') is not True
            or set(search['methods'])!={'random','gp','neural'}
            or any(item.get('trials')!=10 for item in search['methods'].values())):
        raise RuntimeError('Report template requires all 30 live optimizer trials and audited costs')
    hashes=summary.get('input_sha256',{})
    required_files={'results/continuation/search-plan.json','results/measured/budget.jsonl',
        'results/continuation/analysis-spec.json','results/continuation/freeze.json',
        'results/continuation/validation-status.json','results/continuation/independent-validation-evidence.json',
        'results/continuation/cloud-accounting-final.json'}
    required_files.update(f'results/continuation/search/{method}/observations.jsonl' for method in ('random','gp','neural'))
    if not required_files.issubset(hashes):
        raise RuntimeError('Completed analysis lacks required input hashes')
    for relative, expected in hashes.items():
        path=(ROOT/relative).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Analysis input changed or is unavailable: '+relative)
    analyzer=ROOT/'scripts/analyze_continuation.py'
    if summary.get('analyzer_sha256')!=hashlib.sha256(analyzer.read_bytes()).hexdigest():
        raise RuntimeError('Analyzer changed after summary generation; regenerate analysis first')
    required_plots={'live_goodput_vs_allocation.png','live_goodput_vs_trials.png','configuration_goodput_ranges.png',
                    'latency_goodput.png','heldout_accuracy_and_tokens.png'}
    if not required_plots.issubset(summary.get('plot_files',[])) or any(
            not (OUT/'analysis'/name).is_file() or (OUT/'analysis'/name).stat().st_size==0 for name in required_plots):
        raise RuntimeError('Current analysis plots are missing; regenerate analysis with plots before rendering')

def neural_audit(*,make_plot=True):
    history=rows(OUT/'search/neural/observations.jsonl');audit=[];reports=[]
    if len(history)!=10 or len({row['run_id'] for row in history})!=10:
        raise RuntimeError('Four genuine guided proposals require the complete ten-run neural history')
    if any(row.get('source')!='measured' or row.get('partition') not in {'train','validation'} for row in history):
        raise RuntimeError('Neural report requires measured development observations')
    methods=[row.get('optimizer_decision',{}).get('method_used') for row in history]
    if methods!=['shared_initial_design']*6+['neural']*4:
        raise RuntimeError('Expected six initial-design rows followed by four genuinely neural-guided rows; fallback choices do not count')
    model_root=OUT/'search/neural/models'
    available={path.parent.name:path for path in model_root.glob('proposal-*/report.json')}
    if len(available)!=4:
        raise RuntimeError('Exactly four corresponding neural artifact reports are required')
    used=set()
    for ordinal,(i,row) in enumerate(list(enumerate(history))[6:],1):
        decision=row['optimizer_decision'];reference=decision.get('model_artifact',{})
        directory=Path(reference.get('directory','')).name
        if (not re.fullmatch(r'proposal-\d{3}-[0-9a-f]{12}',directory) or directory not in available or directory in used
                or Path(reference.get('report','')).parent.name!=directory or Path(reference.get('report','')).name!='report.json'
                or Path(reference.get('checkpoint','')).parent.name!=directory or Path(reference.get('checkpoint','')).name!='ensemble.pt'):
            raise RuntimeError('Neural decision must identify one unique corresponding artifact directory')
        used.add(directory);path=available[directory];manifest=read(path.with_name('manifest.json'))
        if set(manifest.get('files',{}))!={'ensemble.pt','observations.json','report.json'}:
            raise ValueError('Neural manifest lacks the complete checkpoint/report/observation set')
        for name,sha in manifest['files'].items():
            artifact=path.with_name(name)
            if not artifact.is_file() or artifact.stat().st_size==0 or hashlib.sha256(artifact.read_bytes()).hexdigest()!=sha:
                raise ValueError('Neural model artifact integrity failure')
        report=read(path);provenance=report['provenance'];training=provenance['training_report']
        signature=hashlib.sha256(canonical(provenance)).hexdigest()
        visible=read(path.with_name('observations.json'))
        key=json.dumps([str(row['config_id']),str(row['scenario_id'])],separators=(',',':'))
        if (signature!=manifest.get('proposal_sha256') or signature!=reference.get('proposal_sha256')
                or directory!=f'proposal-{ordinal:03d}-{signature[:12]}'
                or provenance.get('ordinal')!=ordinal or provenance.get('source')!='measured'
                or training.get('source')!='measured'
                or training.get('method_version')!='mlp-ensemble-v2-train-only-log-target-standardization'
                or decision.get('requested_method')!='neural' or decision.get('observations_seen')!=i
                or decision.get('selected_key')!=key or provenance.get('selected_key')!=key
                or provenance.get('selected_prediction')!=decision.get('prediction')
                or provenance.get('acquisition')!=decision.get('acquisition')
                or provenance.get('selected_features')!={name:row[name] for name in ('config_id','scenario_id','config','workload')}
                or visible!=history[:i]
                or provenance.get('observed_sha256')!=hashlib.sha256(path.with_name('observations.json').read_bytes()).hexdigest()
                or training.get('training_records',0)+training.get('validation_records',0)!=i):
            raise ValueError('Neural artifact does not match the actual prior observations and prospective decision')
        members=training.get('members',[])
        if len(members)!=3 or any(not member.get('epochs') or member.get('best_epoch') not in {x.get('epoch') for x in member['epochs']} for member in members):
            raise ValueError('Neural artifact lacks three actual training histories and selected epochs')
        prediction=decision['prediction'];actual=row['metrics']['quality_goodput']
        if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in (prediction['goodput_mean'],actual)):
            raise ValueError('Nonfinite prospective prediction or outcome')
        constant=float(np.mean([r['metrics']['quality_goodput'] for r in history[:i]]))
        audit.append({'run_id':row['run_id'],'observations_seen':i,'config':config(row),
            'prediction':prediction,'observed_quality_goodput':actual,'observed_feasible':row['metrics']['feasible'],
            'artifact_directory':str(path.parent.relative_to(ROOT)),'artifact_sha256':signature,
            'prior_observation_mean':constant,'absolute_error':abs(prediction['goodput_mean']-actual),
            'prior_mean_absolute_error':abs(constant-actual)})
        reports.append(report)
    result={'source':'measured','verified_guided_proposal_count':4,'verified_artifact_report_count':4,
        'prospective_selected_proposals':audit,
        'mean_absolute_error':float(np.mean([x['absolute_error'] for x in audit])),
        'prior_mean_baseline_mae':float(np.mean([x['prior_mean_absolute_error'] for x in audit])),
        'warning':'Only adaptively selected future proposals; not an unbiased whole-space generalization test. Checksums verify retained provenance consistency, not an independent timestamp attestation.'}
    (OUT/'analysis').mkdir(parents=True,exist_ok=True)
    destination=OUT/'analysis/neural-prospective-audit.json'
    destination.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        last=reports[-1];members=last['provenance']['training_report']['members']
        fig,axes=plt.subplots(1,len(members),figsize=(10,3.1),sharey=True)
        for ax,member,index in zip(np.atleast_1d(axes),members,range(1,len(members)+1)):
            h=member['epochs'];ax.plot([x['epoch'] for x in h],[x['train_loss'] for x in h],label='train',color='#367eb8')
            ax.plot([x['epoch'] for x in h],[x['validation_loss'] for x in h],label='internal validation',color='#df8b25')
            ax.axvline(member['best_epoch'],ls=':',color='#777');ax.set(title=f'Ensemble member {index}',xlabel='Epoch');ax.grid(alpha=.2)
        axes=np.atleast_1d(axes)
        axes[0].set_ylabel('Multitask training objective');axes[-1].legend(fontsize=8)
        fig.suptitle('Last prospective neural proposal: actual CPU training histories',fontsize=12);fig.tight_layout()
        for suffix in ('png','svg'):fig.savefig(OUT/'analysis'/('neural-training-histories.'+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig)
    return result,reports

def stability_section():
    path=OUT/'analysis/stability-summary.json'
    if not path.is_file():return ''
    evidence=read(path)
    if evidence.get('source')!='measured' or evidence.get('phase')!='independent_development_validation':
        raise ValueError('Optional stability evidence has the wrong source or phase')
    if evidence.get('status')!='complete':
        return '\nOptional independent-validation queue/memory analysis is '+str(evidence.get('status','unavailable'))+'; no completed stability result is inferred.\n'
    arms=evidence.get('arms',[])
    figure=OUT/'analysis/independent_stability.png'
    if (len(arms)!=2 or {row.get('arm') for row in arms}!={'default','tuned'} or not figure.is_file()
            or 'independent_stability.png' not in evidence.get('plot_files',[])):
        raise ValueError('Completed stability evidence requires both arm records and its plot')
    for row in arms:
        if not row.get('input_sha256'):raise ValueError('Stability input provenance missing')
        for relative,expected in row['input_sha256'].items():
            raw=(ROOT/relative).resolve()
            if ROOT.resolve() not in raw.parents or not raw.is_file() or hashlib.sha256(raw.read_bytes()).hexdigest()!=expected:
                raise ValueError('Stability inputs changed; regenerate the descriptive analysis')
    result=table(['Validation arm','Telemetry samples','Peak client outstanding','Peak engine waiting','Peak sampled GPU GiB'],
        [[row['arm'],row['telemetry_samples'],f(row.get('peak_client_outstanding'),0),
          f(row.get('peak_engine_waiting'),0),f(row['peak_device_memory_bytes']/1024**3,2)
          if row.get('peak_device_memory_bytes') is not None else 'unavailable'] for row in arms])
    return ('\nThe optional independent-validation telemetry audit describes client outstanding requests, engine waiting and sampled device memory in these two measured runs. '
            'Each arm has a finite 150-question cohort with 0.6-per-second spacing through about 248.33 seconds and a quiet tail to 300 seconds. '
            'A draining queue in that tail does not establish stability under indefinite arrivals; this is not a 1,000-request tail-latency study.\n\n'
            +result+'\n\n![Independent-validation queue and sampled-memory trajectories. One finite cohort per arm; no steady-state or P99 guarantee.](results/continuation/analysis/independent_stability.png)\n')

def incumbent_origins(histories, search, selected, *, common_initial_count=6):
    """Attribute observed maxima and finalist to decisions, never method labels alone."""
    lookup={};methods={}
    for method,history in histories.items():
        annotated=[]
        for index,row in enumerate(history):
            decision=row.get('optimizer_decision',{}).get('method_used')
            initial=index<common_initial_count
            if initial!=(decision=='shared_initial_design'):
                raise ValueError('Initial-design position and recorded decision provenance disagree')
            origin={'method':method,'run_id':row['run_id'],'trial':index+1,
                    'phase':'shared_initial_design' if initial else 'adaptive',
                    'decision_method_used':decision,'config_id':row['config_id'],
                    'quality_goodput':row['metrics']['quality_goodput'],
                    'feasible':row['metrics'].get('feasible') is True}
            if row['run_id'] in lookup:raise ValueError('Finalist provenance requires unique live run IDs')
            lookup[row['run_id']]=origin;annotated.append(origin)
        summary=search['methods'][method];incumbent=summary.get('incumbent_run_id')
        origin=lookup.get(incumbent) if incumbent else None
        feasible_rows=[row for row in annotated if row['feasible']]
        actual_incumbent=max(feasible_rows,key=lambda row:row['quality_goodput']) if feasible_rows else None
        if (actual_incumbent and incumbent!=actual_incumbent['run_id']) or (not actual_incumbent and incumbent):
            raise ValueError('Method incumbent must be the first observation attaining its actual feasible maximum')
        if incumbent and (origin is None or origin['method']!=method or not origin['feasible']
            or not math.isclose(origin['quality_goodput'],summary['best_feasible_quality_goodput'],rel_tol=0,abs_tol=1e-12)):
            raise ValueError('Method incumbent does not match its actual feasible history observation')
        initial_values=[r['quality_goodput'] for r in annotated if r['feasible'] and r['phase']=='shared_initial_design']
        adaptive_values=[r['quality_goodput'] for r in annotated if r['feasible'] and r['phase']=='adaptive']
        first=max(initial_values) if initial_values else None
        later=max(adaptive_values) if adaptive_values else None
        tied=[r for r in annotated if r['feasible'] and origin is not None
              and math.isclose(r['quality_goodput'],origin['quality_goodput'],rel_tol=0,abs_tol=1e-12)]
        methods[method]={'incumbent':origin,'best_initial_quality_goodput':first,'best_adaptive_quality_goodput':later,
            'incremental_best_goodput_after_initial':max(0.,later-first) if first is not None and later is not None else None,
            'all_tied_maximum_observations':tied}
    finalist=lookup.get(selected['run_id'])
    if finalist is None or finalist['config_id']!=selected['config_id'] or not finalist['feasible']:
        raise ValueError('Frozen finalist cannot be traced to a feasible observed search decision')
    return {'source':'measured','common_initial_count':common_initial_count,'methods':methods,'selected_finalist':finalist,
        'interpretation':'Method names identify separate live histories. An initial-design winner is not an adaptive discovery; tied later measurements do not increase the incumbent.'}

def discovery_note(origin):
    if origin['phase']=='shared_initial_design':
        return (f"The selected finalist came from trial {origin['trial']} of the {origin['method']} history's shared initial design. "
                "It was not discovered by that method's adaptive acquisition. "
                +("A neural-method winner among common starting configurations is not evidence of neural-guided discovery."
                  if origin['method']=='neural' else 'Its selection does not demonstrate an adaptive-search discovery.'))
    return (f"The selected finalist came from adaptive trial {origin['trial']} of the {origin['method']} history "
            f"(recorded decision: {origin['decision_method_used']}). This provenance alone does not establish superiority over the other optimizers.")

def optional_number(value, *, probability=False):
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Optional result contains a nonfinite or nonnumeric metric')
    if probability and not 0<=value<=1:raise ValueError('Optional accuracy must lie in [0,1]')
    return value

def heldout_interpretation(arms, contrasts):
    """Interpret the saved primary paired contrasts, without presuming a benefit."""
    primary={}
    for row in contrasts:
        if row['endpoint']!='deadline_constrained_accuracy':continue
        if row['name'] in primary:raise ValueError('Duplicated primary held-out contrast')
        primary[row['name']]=row
    names=('tuned_minus_default_short','tuned_minus_default_long','allowance_effect_default',
           'allowance_effect_tuned','serving_by_allowance_interaction')
    if set(primary)!=set(names):raise ValueError('Held-out interpretation needs all five primary contrasts')
    caps=sorted({arm['allowance'] for arm in arms})
    lookup={(arm['arm'],arm['allowance']):arm for arm in arms}
    if len(caps)!=2 or len(lookup)!=4 or set(lookup)!={(name,cap) for name in ('default','tuned') for cap in caps}:
        raise ValueError('Held-out interpretation requires the complete two-by-two design')
    scores={key:optional_number(arm['summary']['deadline_constrained_accuracy'],probability=True)
            for key,arm in lookup.items()}
    short=scores['tuned',caps[0]]-scores['default',caps[0]]
    long=scores['tuned',caps[1]]-scores['default',caps[1]]
    effects=(short,long,scores['default',caps[1]]-scores['default',caps[0]],
             scores['tuned',caps[1]]-scores['tuned',caps[0]],long-short)
    for name,expected in zip(names,effects):
        row=primary[name];estimate=optional_number(row['estimate'])
        ci=[optional_number(value) for value in row['ci']]
        if (len(ci)!=2 or any(value is None for value in ci) or ci[0]>ci[1]
                or estimate is None or not math.isclose(estimate,expected,abs_tol=1e-12)):
            raise ValueError('Held-out primary contrast disagrees with its arm summaries')
    def effect(name):
        row=primary[name]
        return (f"{100*row['estimate']:+.2f} percentage points "
                f"({100*row['confidence']:.0f}% paired interval [{100*row['ci'][0]:.2f}, {100*row['ci'][1]:.2f}])")
    text=(f"The tuned-minus-default deadline-accuracy difference is {effect(names[0])} at {caps[0]:,} tokens "
          f"and {effect(names[1])} at {caps[1]:,} tokens. ")
    if all(primary[name]['ci'][0]<=0<=primary[name]['ci'][1] for name in names[:2]):
        text+='Both configuration-effect intervals include zero, so these observations do not establish a repeatable serving-configuration advantage. '
    text+=(f"Increasing the allowance changes deadline accuracy by {effect(names[2])} for default "
           f"and {effect(names[3])} for tuned. The serving-by-allowance interaction is {effect(names[4])}. ")
    if primary[names[4]]['ci'][0]<=0<=primary[names[4]]['ci'][1]:
        text+='The interaction interval includes zero; it does not establish that tuning uniquely enables the benefit of a larger allowance. '
    if all(math.isclose(arm['summary']['accuracy'],arm['summary']['deadline_constrained_accuracy'],abs_tol=1e-12) for arm in arms):
        text+='Every correct outcome in these saved arm summaries also meets its deadline. '
    return text+'These contrasts count correct answers within the deadline; they do not by themselves identify additional completed requests or a causal deadline-recovery mechanism.'

def replay_ceiling_note(evidence):
    """Check a pre-adaptation ceiling against the artifact's exact source table."""
    coverage=evidence.get('coverage',{});provenance=coverage.get('first_observation_provenance',[])
    arms=evidence.get('arms',[]);initial=evidence.get('initial_points',0)
    if not provenance or not arms or not isinstance(initial,int) or initial<1:return ''
    sources={}
    for method in ('random','gp','neural'):
        path=OUT/'search'/method/'observations.jsonl'
        if path.is_file():
            for row in rows(path):
                if row['run_id'] in sources and row!=sources[row['run_id']]:
                    raise ValueError('Conflicting replay source observations')
                sources[row['run_id']]=row
    if any(p['selected_run_id'] not in sources for p in provenance):
        return 'The static-table source observations needed to verify a pre-adaptation ceiling are unavailable; no saturation claim is inferred.\n\n'
    chosen={}
    for entry in provenance:
        row=sources[entry['selected_run_id']]
        key=json.dumps([row['config_id'],row['scenario_id']],separators=(',',':'))
        if key!=entry['config_scenario_key'] or key in chosen:
            raise ValueError('Replay table provenance contains mismatched or duplicated configurations')
        chosen[key]=row
    source_table=[chosen[key] for key in sorted(chosen)]
    digest=hashlib.sha256(json.dumps(source_table,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if len(source_table)!=coverage.get('observed_configurations') or digest!=evidence.get('table_sha256'):
        raise ValueError('Replay ceiling check does not match the saved source-table hash')
    feasible=[row for row in source_table if row['metrics'].get('feasible') and row['metrics'].get('quality_goodput') is not None]
    if not feasible:return ''
    ceiling=max(optional_number(row['metrics']['quality_goodput']) for row in feasible)
    saturated=[]
    for arm in arms:
        curve=arm.get('curve',[])
        if len(curve)<initial:continue
        before=curve[initial-1].get('best_observed') or {};after=arm.get('best_observed') or {}
        if (before.get('quality_goodput')==ceiling==after.get('quality_goodput')):
            saturated.append(arm)
    if not saturated:return ''
    text=(f"The exact saved source table has a maximum feasible QAG of {f(ceiling)}. "
          f"In {len(saturated)} of {len(arms)} recorded replay arms, the incumbent after the {initial} common starting observations "
          'already equals that table maximum and the final incumbent. ')
    complete={(seed,beta) for seed in evidence.get('seeds',[]) for beta in (0,1)}
    if len(saturated)==len(arms) and {(arm['seed'],arm['beta']) for arm in arms}==complete:
        common=None
        for arm in arms:
            entries={(i,cid,arm['initial_source_run_ids'][i]) for i,cid in enumerate(arm['initial_config_ids'])
                     if sources[arm['initial_source_run_ids'][i]]['metrics'].get('feasible')
                     and sources[arm['initial_source_run_ids'][i]]['metrics'].get('quality_goodput')==ceiling}
            common=entries if common is None else common & entries
        if common:
            position,cid,run_id=sorted(common)[0]
            text+=(f"Every starting design includes configuration `{cid}` from `{run_id}` at position {position+1} "
                   f"(zero-based index {position}). ")
        text+=(f"All {len(evidence['seeds'])} paired replay seeds therefore saturate before adaptive acquisition. "
               'Their equal beta endpoints are uninformative about discovery of the table optimum and provide no evidence that uncertainty has no value. ')
    return text+'This ceiling concerns the retained empirical table, not the unknown population optimum.\n\n'

def uncertainty_ablation_section():
    """Render saved simulated searches only; never run or fill in an ablation."""
    path=OUT/'posthoc-uncertainty-ablation.json'
    heading='### Optional uncertainty ablation: POSTHOC OFFLINE REPLAY\n\n'
    if not path.is_file():
        return heading+'No saved uncertainty-ablation result is available. No replay outcome or hardware finding is inferred.\n'
    evidence=read(path)
    mode='posthoc_offline_replay_over_measured_table'
    if (evidence.get('mode')!=mode or evidence.get('evidence_source')!='measured'
            or evidence.get('is_live_optimizer_experiment') is not False
            or evidence.get('new_hardware_measurements')!=0 or evidence.get('new_accelerator_seconds_charged')!=0):
        raise ValueError('Optional uncertainty result must be explicitly labeled post-hoc replay with zero new hardware charges')
    coverage=evidence.get('coverage',{});arms={};paired={}
    for arm in evidence.get('arms',[]):
        key=(arm['seed'],arm['beta'])
        if arm.get('mode')!=mode or arm['beta'] not in (0,1) or key in arms:
            raise ValueError('Invalid or duplicated optional replay arm')
        arms[key]=arm
    for pair in evidence.get('paired_seed_results',[]):
        if pair['seed'] in paired:raise ValueError('Duplicated optional replay seed summary')
        paired[pair['seed']]=pair
    seeds=list(evidence.get('seeds',[]))
    if len(set(seeds))!=len(seeds) or any(seed not in seeds for seed,_ in arms) or any(seed not in seeds for seed in paired):
        raise ValueError('Optional replay rows differ from the declared seed list')
    values=[];details=[]
    for seed in seeds:
        one,zero=arms.get((seed,1)),arms.get((seed,0));pair=paired.get(seed,{})
        status=pair.get('status','partial_arm_records' if one or zero else 'no_recorded_replay')
        scores=[optional_number(arm.get('best_observed',{}).get('quality_goodput'))
                if arm and arm.get('best_observed') is not None else None for arm in (one,zero)]
        delta=scores[0]-scores[1] if all(value is not None for value in scores) else None
        if one and zero:
            if (one.get('initial_config_ids')!=zero.get('initial_config_ids')
                    or one.get('initial_source_run_ids')!=zero.get('initial_source_run_ids')
                    or len(one.get('initial_config_ids',[]))!=evidence.get('initial_points')):
                raise ValueError('Optional beta comparison lacks paired common initial evidence')
        if status=='completed':
            if not one or not zero or pair.get('same_initial_configurations') is not True or pair.get('same_initial_source_outcomes') is not True:
                raise ValueError('Completed optional replay seed lacks two paired arms')
            expected=(scores[0],scores[1],delta)
            actual=[pair.get(key) for key in ('beta1_best_observed_goodput','beta0_best_observed_goodput','beta1_minus_beta0_goodput')]
            if any((left is None)!=(right is None) or (left is not None and not math.isclose(left,optional_number(right),abs_tol=1e-12))
                   for left,right in zip(expected,actual)):
                raise ValueError('Optional replay seed summary disagrees with its recorded arm outcomes')
        adaptive=' / '.join(str(arm.get('adaptive_trials','unavailable')) if arm else 'unavailable' for arm in (one,zero))
        debits=' / '.join(f(optional_number(arm.get('replay_debit_seconds')),2) if arm else 'unavailable' for arm in (one,zero))
        values.append([seed,status,f(scores[0]),f(scores[1]),f(delta),adaptive,debits])
        if pair.get('reason'):details.append(f"Seed {seed}: {pair['reason']}")
    status=evidence.get('status','unavailable')
    text=(heading+f"Saved replay status: **{status}**. Static-table coverage is {coverage.get('observed_configurations','unavailable')} "
          f"of {coverage.get('expected_configurations','unavailable')} frozen configurations ({coverage.get('status','unavailable')}). "
          'Each available table entry is the first chronological live observation for that configuration. Missing configurations are excluded, not imputed.\n\n')
    if values:
        text+=table(['Replay seed','Recorded status','Best QAG beta=1','Best QAG beta=0','Difference','Adaptive choices 1 / 0','Replay debit seconds 1 / 0'],values)+'\n\n'
    else:text+='No paired seed results were recorded.\n\n'
    if details:text+=' '.join(details)+'\n\n'
    text+=replay_ceiling_note(evidence)
    text+=('The QAG values above are outcomes revealed from a static measured table along simulated search paths. '
           'The differences describe beta sensitivity conditional on that noisy, adaptively collected table; they are not new hardware measurements, '
           'a live optimizer ranking or independent repetitions. Replay debits sum existing source costs and are never added to the global real budget. '
           'Insufficient coverage can reduce the requested four adaptive choices, and invalid grouped splits can skip a seed. '
           'There is no across-hardware confidence interval from these replay seeds.\n\n'
           f"Source: `results/continuation/{path.name}`; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`.\n")
    return text

def unseen_context_section():
    """Render actual optional measured arms, preserving incomplete/failed states."""
    path=OUT/'unseen-context-summary.json'
    heading='### Optional measured longer-context diagnostic\n\n'
    if not path.is_file():
        return heading+'No saved longer-context summary is available. Preregistration alone does not establish that either optional arm ran.\n'
    evidence=read(path);plan_path=OUT/'unseen-context-plan.json'
    if not plan_path.is_file():raise ValueError('Optional context result lacks its preregistered plan')
    plan=read(plan_path);proof=dict(plan);plan_hash=proof.pop('plan_hash',None)
    actual_hash=hashlib.sha256(json.dumps(proof,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if (not plan_hash or actual_hash!=plan_hash or evidence.get('plan_hash')!=plan_hash
            or evidence.get('secondary_only') is not True or evidence.get('core_selection_unchanged') is not True
            or evidence.get('official_test_data_accessed') is not False):
        raise ValueError('Optional context provenance or declared secondary scope is invalid')
    arms=evidence.get('arms',[]);lookup={};values=[];cost=0.
    for arm in arms:
        name=arm.get('arm')
        if name not in {'default','tuned'} or name in lookup or arm.get('source')!='measured':
            raise ValueError('Optional context arms must be unique measured default/tuned observations')
        lookup[name]=arm
        count=arm.get('measurement_cohort_size');outcomes=arm.get('measurement_outcomes',[])
        if not isinstance(count,int) or isinstance(count,bool) or count<=0 or len(outcomes)!=count or len({r['question_id'] for r in outcomes})!=count:
            raise ValueError('Optional context result lacks its complete measurement denominator')
        if [r['question_id'] for r in outcomes]!=plan.get('measurement_question_ids'):
            raise ValueError('Optional context outcomes differ from the frozen measurement cohort')
        for metric,field in (('full_measurement_cohort_accuracy','correct'),('full_measurement_deadline_accuracy','deadline_correct')):
            number=optional_number(arm.get(metric),probability=True)
            if number is None or not math.isclose(number,sum(bool(r.get(field)) for r in outcomes)/count,abs_tol=1e-12):
                raise ValueError('Optional context accuracy differs from its full measurement cohort')
        charge=optional_number(arm.get('cost',{}).get('accelerator_seconds'))
        if charge is None or charge<0:raise ValueError('Optional measured context arm lacks an actual nonnegative cost')
        cost+=charge
        values.append([name,arm.get('execution_status','unavailable'),count,
                       pct(arm['full_measurement_cohort_accuracy']),pct(arm['full_measurement_deadline_accuracy']),
                       f(optional_number(arm.get('metrics',{}).get('quality_goodput'))),f(charge,3),
                       arm.get('missing_measurement_records','unavailable')])
    saved_cost=optional_number(evidence.get('actual_accelerator_seconds'))
    if saved_cost is None or not math.isclose(cost,saved_cost,abs_tol=1e-6):
        raise ValueError('Optional context total cost differs from its recorded arms')
    status=evidence.get('status','unavailable')
    if status=='not_run' and arms:raise ValueError('A not-run optional context summary cannot contain measured arms')
    if status in {'completed','completed_with_failed_runs'}:
        if set(lookup)!={'default','tuned'}:raise ValueError('Completed optional context summary requires both measured arms')
        all_ok=all(arm.get('execution_status')=='ok' for arm in arms)
        if (status=='completed')!=all_ok:raise ValueError('Optional context completion label disagrees with arm execution status')
    if set(lookup)=={'default','tuned'}:
        pairs=[[(r['question_id'],r.get('sampling_seed')) for r in lookup[name]['measurement_outcomes']]
               for name in ('default','tuned')]
        if pairs[0]!=pairs[1]:raise ValueError('Optional context arm question/seed pairing differs')
    text=heading+f"Saved execution status: **{status}**. "
    if evidence.get('reason'):text+='Recorded reason: '+str(evidence['reason'])+'. '
    text+=f"Available measured arms: {len(arms)}/2; recorded arm allocation: {f(cost,3)} accelerator-seconds.\n\n"
    if values:
        text+=table(['Arm','Execution','Cohort','Accuracy','Deadline accuracy','Window QAG','Actual GPU-s','Missing records'],values)+'\n\n'
        missing={'default','tuned'}-set(lookup)
        if missing:text+='Unexecuted or unavailable arm(s): '+', '.join(sorted(missing))+'. No paired effect is inferred.\n\n'
    else:text+='No measured optional arm outcome is reported.\n\n'
    contrasts=[];notes=[]
    for key,label,metric in (('paired_accuracy_difference','Accuracy','full_measurement_cohort_accuracy'),
                             ('paired_deadline_accuracy_difference','Deadline accuracy','full_measurement_deadline_accuracy')):
        contrast=evidence.get(key)
        if not contrast:continue
        if set(lookup)!={'default','tuned'}:raise ValueError('Optional paired interval requires both measured arms')
        difference=optional_number(contrast.get('difference_candidate_minus_reference'))
        expected=lookup['tuned'][metric]-lookup['default'][metric]
        if difference is None or not math.isclose(difference,expected,abs_tol=1e-12):
            raise ValueError('Optional paired difference disagrees with measured arm accuracy')
        interval=contrast.get('ci');confidence=optional_number(contrast.get('confidence'),probability=True)
        if interval is None:interval_text='unavailable'
        elif len(interval)!=2 or any(optional_number(value) is None for value in interval) or interval[0]>interval[1]:
            raise ValueError('Optional paired interval is malformed')
        else:interval_text=f'[{f(100*interval[0],2)}, {f(100*interval[1],2)}]'
        contrasts.append([label,f(100*difference,2),interval_text,pct(confidence),contrast.get('question_clusters','unavailable')])
        if contrast.get('warning'):notes.append(str(contrast['warning']))
        if contrast.get('method'):notes.append('Recorded method: '+str(contrast['method'])+'.')
    if contrasts:
        text+=table(['Paired tuned minus default','Difference, pp','Descriptive interval, pp','Confidence','Question clusters'],contrasts)+'\n\n'
    elif len(arms)==2:text+='No paired confidence interval was saved; none is fabricated from arm point estimates.\n\n'
    if notes:text+=' '.join(dict.fromkeys(notes))+'\n\n'
    text+=('This secondary diagnostic uses unused development questions with longer-context prompts and the original frozen selections. '
           'It does not access official test data or change the core conclusions. Failed attempts remain labeled; an arm row does not imply successful serving. '
           'Missing requests remain incorrect in the full cohort. Window QAG and full-cohort/deadline accuracy are distinct endpoints. '
           'One run per arm and 54 measured questions do not establish robust P99, population noninferiority or broad workload transfer.\n\n'
           f"Source: `results/continuation/{path.name}`; SHA-256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`; plan `{plan_hash}`.\n")
    return text

def terminal_accounting_prose(terminal, global_seconds):
    """Keep charged upper bounds distinct from unknown actual provider end times."""
    upper=terminal.get('duration_basis')=='terminal_observation_upper_bound'
    total=f(global_seconds/3600,4)
    if upper:
        stage=terminal.get('terminal_status',{}).get('stage')
        if (terminal.get('charge_upper_bound') is not True or stage not in {'CANCELED','ERROR','COMPLETED'}
                or any(key not in terminal or terminal[key] is not None
                       for key in ('provider_finished_at','finished_at','actual_accelerator_seconds'))):
            raise RuntimeError('Upper-bound accounting must preserve unknown actual finish and confirm terminal status')
        status='confirmed canceled' if stage=='CANCELED' else 'confirmed in terminal state '+stage
        endpoint=terminal['accounting_end_at']
        return {'upper_bound':True,'total_label':'TOTAL CHARGED (upper bound)',
            'summary':f'Total task charge, including all prior failures and cloud overhead, is a conservative upper bound of {total} accelerator-hours, below the original eight-hour ceiling. The cloud job is {status}; its actual finish timestamp and exact allocation duration remain unknown.',
            'detail':f'The provider reports the job as {status}, but supplies neither a finish timestamp nor a duration. The accounting endpoint is the recorded terminal observation at {endpoint}; it is not represented as the actual job end. The charged creation-to-observation interval is {f(terminal["charged_accelerator_seconds"],3)} accelerator-seconds, a conservative upper bound. All cloud time through that observation, including downloads, setup, CPU optimization, waiting and artifact export, is covered once; per-trial charges are subtracted before residual overhead is appended. The global ledger equals prior allocations plus this upper-bound charge. Actual allocation could be shorter.',
            'reproduction':f'The global ledger reconciles to prior charges plus the conservative creation-to-terminal-observation upper bound. The cloud job is {status} before reporting completion; the provider finish timestamp and exact actual duration remain unknown.'}
    return {'upper_bound':False,'total_label':'TOTAL',
        'summary':f'Total task allocation, including all prior failures and cloud overhead, is {total} accelerator-hours, below the original eight-hour ceiling.',
        'detail':'All elapsed cloud allocation, including downloads, setup, CPU optimization, waiting and artifact export, is charged once using provider creation and finish timestamps; per-trial costs are subtracted before residual overhead is appended.',
        'reproduction':'The global ledger and provider creation/finish timestamps reconcile to recorded precision; the live job has ended before reporting completion.'}


def optimizer_resource_section(search):
    timing=search.get('optimizer_timing');cap=search.get('method_cap_audit')
    if timing is None or cap is None:
        raise ValueError('Regenerate continuation analysis to obtain separate optimizer timers and method-cap audit')
    timers=[];caps=[]
    for method,values in search['methods'].items():
        timers.append([method,f(values['optimizer_cpu_seconds']),f(values['optimizer_wall_seconds']),
            values.get('missing_optimizer_wall_time_count',0)])
        audit=cap['methods'][method]
        excess=audit.get('fully_attributed_excess_seconds')
        caps.append([method,f(audit['base_trial_accelerator_seconds'],2),
            f(audit['fully_attributed_accelerator_seconds'],2),f(audit['declared_base_trial_admission_cap_seconds'],2),
            'unavailable' if excess is None else format(excess,'.6g'),audit['status'].replace('_',' ')])
    text=(table(['Method','Policy CPU seconds','Policy wall seconds','Missing wall-timer records'],timers)
        +'\n\nRecorded policy CPU time totals '+f(timing['optimizer_cpu_seconds_sum'])+' seconds; '
        'recorded policy wall time totals '+f(timing['optimizer_wall_seconds_sum'])+' seconds. '
        'These are the process-CPU and elapsed-wall timers around `SearchPolicy.choose`, including fitting, acquisition and artifact work inside that call. '
        'They exclude tokenizer/setup work, policy loading and later policy saves outside the timed call. '
        'Shared initial-design rows record zero policy-selection time. The global ledger can show zero optimizer CPU annotations because it is written before the per-method timing enrichment; '
        'this does not mean optimizer computation was absent. CPU time and elapsed wall time are separate clocks, and neither is added again to GPU allocation. '
        'Reserved GPU wall during fitting and other intervening work is already charged once through the method-tagged cloud-overhead events.\n\n'
        +table(['Method','Base trial GPU-s','Attributed GPU-s','Base admission amount, s','Attributed excess, s','Audit status'],caps)
        +'\n\nThe configured '+f(cap['declared_base_trial_admission_cap_seconds'],0)+'-second amount is used by the live search admission logic against base trial accelerator charges. '
        'The GPU overhead between trials is reconciled and attributed afterward. The table therefore compares the full attributed total with the same declared amount as a separate audit; '
        'it does not claim that the code enforced that amount as an end-to-end method resource ceiling. '
        'Any excess remains visible without altering measured outcomes or the experiment-completion label. The original eight-hour global ceiling remains the resource constraint.\n')
    return text

def report_figures(summary):
    """Present the audited results with plain labels; do not alter analysis data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    palette={'random':'#3174a1','gp':'#bc702a','neural':'#248577'}
    labels={'random':'Random search','gp':'Gaussian process','neural':'Neural ensemble'}
    with plt.rc_context({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.labelcolor':'#263447','text.color':'#263447'}):
        fig,ax=plt.subplots(figsize=(9.8,4.0))
        for method in ('random','gp','neural'):
            curve=summary['search']['curves'][method]
            ax.step([r['cumulative_accelerator_seconds']/60 for r in curve],
                    [r['best_feasible_goodput'] for r in curve],where='post',
                    color=palette[method],label=labels[method],linewidth=2,
                    marker={'random':'o','gp':'s','neural':'D'}[method],markersize=4)
        ax.axhline(summary['default_reference']['metrics']['quality_goodput'],
                   color='#6a737c',linestyle='--',label='Default reference',linewidth=1.4)
        ax.set(xlabel='GPU minutes used by each search method',
               ylabel='Best correct, timely answers per second',
               title='Live search progress',xlim=(3,52),ylim=(.335,.452))
        ax.set_yticks([.34,.36,.38,.40,.42,.44]);ax.grid(axis='y',alpha=.2)
        ax.legend(loc='upper left',ncol=2,fontsize=10,frameon=False)
        fig.tight_layout()
        for suffix in ('png','svg'):
            fig.savefig(OUT/'analysis'/('report-search-progress.'+suffix),dpi=190,bbox_inches='tight')
        plt.close(fig)
        lookup={(x['arm'],x['allowance']):x['summary'] for x in summary['heldout']['arms']}
        fig,axes=plt.subplots(1,2,figsize=(10,4.0))
        x=np.arange(2);width=.34
        for name,offset,color in (('default',-width/2,'#3174a1'),('tuned',width/2,'#248577')):
            values=[lookup[(name,limit)] for limit in (2048,4096)]
            if any(v['generated_tokens'] is None for v in values):
                raise RuntimeError('Report token chart requires complete measured token usage')
            accuracy=[100*v['deadline_constrained_accuracy'] for v in values]
            bars=axes[0].bar(x+offset,accuracy,width,color=color,label=name.title())
            axes[0].bar_label(bars,labels=[f"{n:.1f}%\n{round(v['accuracy']*v['cohort_size'])}/{v['cohort_size']}"
                                         for n,v in zip(accuracy,values)],padding=4,fontsize=9)
            bars=axes[1].bar(x+offset,[v['generated_tokens']/1000 for v in values],width,
                             color=color,label=name.title())
            axes[1].bar_label(bars,labels=[f"{v['generated_tokens']/1000:.1f}" for v in values],
                              padding=4,fontsize=9)
        for ax in axes:
            ax.set_xticks(x,['2,048 tokens','4,096 tokens'])
            ax.set_xlabel('Output allowance per answer')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes[0].set(title='Correct answers by the deadline',ylabel='Accuracy (% of 128 questions)',ylim=(0,107))
        axes[1].set(title='Total generated output',ylabel='Output tokens (thousands)',ylim=(0,255))
        axes[0].legend(loc='upper left',ncol=2,fontsize=9,frameon=False)
        axes[1].legend(loc='upper left',ncol=2,fontsize=9,frameon=False)
        fig.tight_layout(w_pad=3)
        for suffix in ('png','svg'):
            fig.savefig(OUT/'analysis'/('report-heldout-results.'+suffix),dpi=190,bbox_inches='tight')
        plt.close(fig)


def main():
    a=read(OUT/'analysis/summary.json')
    verify_report_evidence(a)
    study=read(ROOT/'configs/study-cloud.json')
    plan=read(OUT/'search-plan.json')
    freeze=read(OUT/'freeze.json')
    if study!=freeze['study']:
        raise RuntimeError('Current study differs from the validated frozen protocol')
    selected=next(x for x in freeze['selected'] if not x.get('is_default'))
    dm=a['default_reference']['metrics']; search=a['search']; acc=a['accounting']
    val=a['independent_validation']['evidence']
    ci=val['tuned']['paired_accuracy_ci']; exact=val['tuned']['paired_noninferiority']
    arms=a['heldout']['arms']; contrasts=a['heldout']['preregistered_paired_contrasts']
    terminal=acc['terminal_cloud']['record']
    terminal_accounting_prose(terminal,acc['global_spent_accelerator_seconds'])
    audit,reports=neural_audit(make_plot=False)
    origins=incumbent_origins(
        {method:rows(OUT/'search'/method/'observations.jsonl') for method in search['methods']},
        search,selected,common_initial_count=plan['common_initial_count'])
    (OUT/'analysis/incumbent-provenance.json').write_text(
        json.dumps(origins,indent=2,sort_keys=True,allow_nan=False)+'\n')
    # Preserve the existing evidence checks while keeping their detailed audit
    # prose out of the study narrative. The full records remain in the bundle.
    stability_section()
    uncertainty_ablation_section()
    unseen_context_section()
    context=read(OUT/'unseen-context-summary.json')
    context_arms={x['arm']:x for x in context['arms']}
    context_ci=context['paired_deadline_accuracy_difference']['ci']
    charge=terminal.get('estimated_charge_usd',terminal.get('estimated_total_charge_usd'))
    labels={'random':'Random search','gp':'Gaussian process','neural':'Neural ensemble'}
    method_table=[]
    for method in ('random','gp','neural'):
        m=search['methods'][method]; origin=origins['methods'][method]
        method_table.append([labels[method],f(origin['best_initial_quality_goodput']),
            f(m['best_feasible_quality_goodput']),
            f(origin['incremental_best_goodput_after_initial']),
            f(m['total_attributed_accelerator_seconds']/60,2)])
    prospective_table=[[x['config'].replace('batch ','').replace(', seq ',' / '),
        f(x['prediction']['goodput_mean']),f(x['observed_quality_goodput'])]
        for x in audit['prospective_selected_proposals']]
    validation_table=[]
    for name in ('default','tuned'):
        m=val[name]['metrics']; count=val[name]['question_count']
        validation_table.append([name.title(),f"{round(m['accuracy']*count)}/{count}",
            pct(m['accuracy']),f(m['quality_goodput'])])
    final_table=[]
    for arm in arms:
        z=arm['summary']
        tokens=f"{z['generated_tokens']:,}" if z['generated_tokens'] is not None else 'unavailable'
        final_table.append([arm['arm'].title(),f"{arm['allowance']:,}",
            f"{round(z['accuracy']*z['cohort_size'])}/{z['cohort_size']}",
            pct(z['deadline_constrained_accuracy']),tokens])
    contrast_labels={
        'tuned_minus_default_short':'Tuned minus default, 2,048 tokens',
        'tuned_minus_default_long':'Tuned minus default, 4,096 tokens',
        'allowance_effect_default':'4,096 minus 2,048 tokens, default',
        'allowance_effect_tuned':'4,096 minus 2,048 tokens, tuned',
        'serving_by_allowance_interaction':'Difference between the two allowance effects'}
    contrast_table=[[contrast_labels[r['name']],f(100*r['estimate'],2),
        f"[{f(100*r['ci'][0],2)}, {f(100*r['ci'][1],2)}]"]
        for r in contrasts if r['endpoint']=='deadline_constrained_accuracy']
    report_figures(a)
    text=f'''# Tuning vLLM for Small-Model Reasoning Under a Fixed GPU-Time Budget

**Experiments: 3 November 2026 | Report revised: 4 November 2026**

## Abstract

This study tested whether a small neural network could choose vLLM serving settings that produce more correct reasoning answers within a fixed compute budget. We served Qwen3-1.7B on one NVIDIA A10G GPU and compared neural-guided search with random search and Gaussian-process optimization. The default configuration provided a usable baseline of {f(dm['quality_goodput'])} correct, timely answers per second. Across 30 live trials, neural and random search both reached 0.433, while the Gaussian process reached 0.400. The neural result came from a common starting configuration; its four learned choices did not improve that result.

After independent validation on 150 questions, the selected settings were tested on 128 held-out questions. Each condition received the same six-minute GPU allocation. The tuned settings produced seven more correct answers than default at both output limits, but the uncertainty intervals included no difference. Raising the output limit from 2,048 to 4,096 tokens produced 13 more correct answers under both configurations. The study therefore found a benefit from allowing longer answers in this sample, but did not establish that neural tuning was better than the alternatives or uniquely enabled that benefit. Total resource use was conservatively bounded at {f(acc['global_spent_accelerator_seconds']/3600,3)} accelerator-hours, within the eight-hour limit.

## 1. Research question

Serving settings determine how an inference engine shares GPU capacity among requests. A setting that generates tokens quickly may still be unhelpful if answers are wrong, unfinished or too late. This study therefore asked two questions: can a learned performance model find settings that return more correct answers on time, and do the selected settings improve reasoning results when total GPU time is held fixed?

The study builds on work on serving autotuning, including SCOOT, and neural models for expensive optimization, including DNGO [1, 2]. Its focus is the combination of answer correctness, response speed and a limited experiment budget. The Qwen language-model weights remained unchanged throughout. Only the small performance model used to select serving settings was trained.

## 2. Experimental design

### Model, hardware and serving settings

All main measurements used Qwen3-1.7B in thinking mode, vLLM 0.10.2 and one NVIDIA A10G GPU [3, 4]. The model and tokenizer versions were fixed before the comparison. The following settings were shared across methods.

| Setting | Value |
| --- | --- |
| Model precision and devices | BF16; one GPU |
| Maximum context | 8,192 tokens, including input and output |
| Output limit during search | 4,096 tokens per answer |
| Sampling | Temperature 0.6; top-p 0.95; top-k 20; min-p 0; presence penalty 1.5 |
| Engine controls | Chunked prefill enabled; prefix caching disabled; GPU memory fraction 0.8 |
| Settings searched | Batched-token limit: 512, 1,024 or 2,048; concurrent-sequence limit: 8, 16, 32 or 64 |

The two searched settings form 12 possible configurations. The batched-token limit controls how many tokens the scheduler can process in a step; the sequence limit controls how many requests it can process concurrently. The default comparison let vLLM choose these two limits while retaining the same model, decoding and other engine settings. Here, "default" refers to these scheduler limits, rather than every vLLM option.

Development pilots led to the fixed presence penalty of 1.5 to reduce repetitive output. The 4,096-token limit was a study budget choice; it does not represent the model's maximum reasoning capacity. The earlier local feasibility runs were excluded from the cloud comparison and from training the performance model.

### Questions and data separation

We used GSM8K, a benchmark of grade-school mathematics questions [5]. Training questions supplied the search workload and independent validation. Official test questions were reserved for the final comparison. Expected answers were available only to the scorer and were never included in generation prompts or performance-model inputs.

| Stage | Questions and purpose |
| --- | --- |
| Baseline and live search | The same 54 scored training questions per run; two additional requests for warmup and ramp-up |
| Independent validation | 150 different training questions, used after the serving configuration was selected |
| Held-out test | The same 128 official test questions in all four conditions |
| Longer-context check | 54 other training questions with an added irrelevant prefix |

Question order and generation seeds were matched within each comparison. The test-question order was fixed with seed 724 before its contents were fetched, and generation seeds were 724-851. The selected serving configuration and the two output limits were fixed before the held-out test. These controls reduce accidental use of test results for tuning, although prior exposure to GSM8K during the language model's pretraining cannot be ruled out.

### What counts as a successful answer

The primary search measure was **quality-adjusted goodput**, abbreviated QAG: the number of correct answers that finished within the measurement window and met all response-time limits, divided by the window length in seconds. Higher values mean more useful answers delivered on time.

The response-time limits were five seconds to the first output, an average of 0.10 seconds per output token after the first, and 180 seconds from dispatch to completion. Timing was measured at the client. The per-token measure used total output tokens and elapsed time; streaming chunks do not reveal the exact timing of every token.

Scoring required a finished response with a valid final numerical answer. Truncated output, unfinished reasoning, invalid answers, timeouts and unanswered requests counted as incorrect. Accuracy used every scored question in the denominator. Answers completed after the measurement window could improve full-cohort accuracy but could not improve QAG. A candidate also had to stay within a measured accuracy drop of two percentage points from the baseline and an error rate of 1%.

During search, requests arrived at 0.6 per second over a 90-second measurement window, following five seconds each of warmup and ramp-up. A request could continue into a drain period, subject to the 120-second timeout. Arrival schedules included small fixed jitter and were matched across methods at the same trial number. The same questions were reused, so this stage measured performance on one repeated workload.

## 3. Search methods and fair comparison

Each method evaluated the same six starting configurations, followed by four choices made using its own previous results. The starting configurations were measured separately for each method and charged to that method's budget. The order of methods rotated between rounds to reduce the influence of changes over time. There were ten live trials per method and 30 in total, using one search seed.

**Random search** chose uniformly among untried configurations. **Gaussian-process optimization** fitted a conventional statistical performance model and selected configurations using expected improvement, weighted by predicted feasibility. **Neural-guided search** trained three small feed-forward networks, each with hidden layers of 64 and 32 units. Their inputs described the serving configuration and known workload; their outputs predicted QAG, 99th-percentile response time and the probability of satisfying the constraints.

The neural networks were trained by gradient descent with AdamW. Feature scaling and target normalization used only training observations. Related records were kept together when separating training and internal validation data, and early stopping selected the saved weights. Each proposal used only results already measured by that method. The actual weights, visible observations and predictions were saved before the next live trial.

The neural selection score was the predicted probability of feasibility multiplied by the sum of predicted goodput and the ensemble's standard deviation. This gives some preference to promising settings about which the networks disagree. The disagreement is a heuristic measure of uncertainty, not a calibrated confidence interval. The response-time prediction was an auxiliary training target and did not add a separate selection penalty in this experiment.

Every trial reserved approximately 300 GPU-seconds, including startup, measurement, shutdown and unused time. Each method had the same ten-trial limit and 3,175-second trial admission budget. The results below include its attributed overhead as well as its trial time; all three totals remained below that amount. Time spent fitting models while the GPU was reserved was included in resource accounting.

## 4. Baseline and live search results

The default reference completed all 54 scored questions: 45 were correct, three were truncated and no request failed at the transport level. Of the correct answers, 32 finished within the measurement window and met the response-time limits. This gave {f(dm['quality_goodput'])} correct, timely answers per second and an overall accuracy of {pct(dm['accuracy'])}.

| Search method | Best starting QAG | Best final QAG | Improvement after starting trials | GPU minutes, including attributed overhead |
| --- | --- | --- | --- | --- |
{chr(10).join('| '+' | '.join(str(v) for v in row)+' |' for row in method_table)}

QAG values are correct, timely answers per second. Random search found its best result on trial 10. The Gaussian process and neural ensemble found their best results among the six starting trials. The four neural-guided choices therefore added no improvement over the neural method's starting result. All 30 trials met the measured feasibility constraints.

![Figure 1. Best measured quality-adjusted goodput as each search method spends GPU time. Shared setup time is excluded from the method curves but included in the total study budget.](results/continuation/analysis/report-search-progress.png)

The final costs differed by less than five seconds. At the lowest common completed cost, about 50.08 GPU minutes, the neural history had reached 0.433 and the other methods 0.400; random search's tenth result was just beyond that point. This boundary effect, together with variation between repeated measurements of the same starting settings, makes a strong ranking inappropriate. One search per method is insufficient to establish which optimizer would perform best across repeated searches.

The selected configuration used a batched-token limit of {selected['config']['max_num_batched_tokens']:,} and a sequence limit of {selected['config']['max_num_seqs']}. It was chosen by the predefined rule: highest feasible measured QAG, then configuration identifier and method/run order for ties. Its winning observation was the sixth starting trial in the neural history. This is evidence about that observed configuration, rather than a configuration discovered by the learned search.

### Did the neural model predict performance well?

The four predictions below were saved before the corresponding live measurements. They provide a direct check of the model's usefulness for the choices it actually made.

{table(['Batched-token / sequence limits','Predicted QAG','Measured QAG'],prospective_table)}

The neural model's average absolute prediction error was {f(audit['mean_absolute_error'])} correct answers per second. Simply predicting the mean of its previous observations had a smaller error of {f(audit['prior_mean_baseline_mae'])}. These four selected trials are too few to characterize general prediction accuracy, but they provide no evidence that the learned model helped this search. Each fit had only 6-9 observations and one or two internal validation groups. All observed trials were feasible, so the model was also not tested on distinguishing feasible from infeasible settings.

## 5. Independent quality validation

The selected configuration and default were then compared on 150 unused training questions with matched generation seeds. Requests were spaced at 0.6 per second for about 248 seconds within a 300-second measurement window. This differs from the search workload, so its QAG values should be compared within this table.

{table(['Configuration','Correct answers','Accuracy','QAG, correct/s'],validation_table)}

Both runs completed all 150 requests without transport errors. Tuned accuracy was 2.67 percentage points higher, with a paired 95% bootstrap interval from -1.33 to 6.67 points. The predefined empirical rule allowed the held-out test to proceed because the measured accuracy drop was within two points. However, the conservative statistical check had a lower bound of -5.09 points: it could not rule out a true drop larger than the allowed two points. Passing the practical execution rule therefore did not prove that tuning preserved accuracy. The candidate and sample size were not changed after these results.

## 6. Held-out test under equal GPU allocation

The final experiment crossed two serving configurations, default and tuned, with two output limits, 2,048 and 4,096 tokens. All four conditions used the same 128 official test questions and matched generation seeds. Each received a nominal 360-second allocation of one GPU, including startup, a 30-second teardown allowance and any unused time. Actual allocations differed by less than 0.00002 seconds.

All questions were scheduled together once the server was ready. Startup reduced the usable response deadline to approximately 291-294 seconds. This tests a finite batch of questions within a whole-device time budget. It does not measure how the service behaves under indefinitely continuing arrivals.

{table(['Configuration','Output limit, tokens','Correct answers','Accuracy by deadline','Total output tokens'],final_table)}

Every condition returned all 128 responses before its deadline, with no transport errors. Reaching the output limit still counted as a scoring failure: the default runs had 25 and 10 truncated answers at the short and long limits, while the tuned runs had 24 and 10. The seven-answer advantage for the tuned configuration therefore came from differences in answer correctness, not from completing more requests before the deadline.

![Figure 2. Accuracy and total generated tokens for the four held-out conditions. Every condition used the same questions and six-minute GPU allocation.](results/continuation/analysis/report-heldout-results.png)

{table(['Comparison','Difference, percentage points','95% paired interval, points'],contrast_table)}

At either output limit, the tuned configuration produced seven additional correct answers, a gain of 5.47 percentage points. Both intervals include zero, so the experiment does not establish a repeatable serving-configuration advantage. Matching random seeds did not force identical generated answers across scheduling configurations; this design does not identify the cause of those differences.

Raising the output limit produced 13 additional correct answers, a gain of 10.16 points, under each configuration. The intervals for these two allowance effects were above zero. The difference between the allowance effects was zero, with a wide interval. Thus, the results support a benefit from the larger output allowance in this sample, without evidence that the tuned settings uniquely enabled it.

Intervals used 2,000 paired bootstrap resamples of questions with seed 724. Each interval describes one comparison separately. They reflect variation across the sampled questions, not variation across repeated deployments. The four conditions were run once each in a fixed order: default-short, default-long, tuned-short and tuned-long.

## 7. Additional checks and limitations

A separate longer-context check added an irrelevant prefix to 54 unused training questions and retained the frozen serving settings. Both configurations answered 48 correctly. Default QAG was {f(context_arms['default']['metrics']['quality_goodput'])} and tuned QAG was {f(context_arms['tuned']['metrics']['quality_goodput'])}. The difference in the fraction of questions answered correctly within the window was 5.56 percentage points, with a 95% interval from {f(100*context_ci[0],2)} to {f(100*context_ci[1],2)}. This small check did not establish an advantage on longer inputs.

A CPU-only replay compared neural selection with and without its uncertainty bonus. In every replay, the starting configurations already included the best entry in the recorded table. The equal final results therefore could not show whether the uncertainty bonus helps discover good settings. This replay was not an additional live GPU experiment.

The main limits are the small search space, one model, one GPU, one search per method and few observations for neural training. Reusing the profiling questions means that internal model validation did not test transfer to new question content. Selecting the highest short-run result can also favor measurement noise; independent validation helps check that result but does not remove all uncertainty.

Only 54 questions were measured per search trial, and each held-out condition was run once. These samples do not support reliable claims about rare slow responses or long-running queue stability. Fixed test order may also allow changes in hardware state or caches to affect results. The study did not evaluate quantization, multiple GPUs, separate prefill/decode servers or transfer to other hardware.

## 8. Resource use, reproducibility and conclusion

The entire task used a conservative upper bound of **{f(acc['global_spent_accelerator_seconds']/3600,3)} accelerator-hours**, including earlier development, unsuccessful launches, setup, search, validation, idle reservations and result export. This is below the original eight-hour budget. Estimated Hugging Face charges were **US${f(charge,4)}**, below the authorized US$10. The provider confirmed cancellation but did not supply an exact finish time, so accounting extended to the recorded cancellation observation. The cost is a conservative estimate, not a verified invoice.

The accompanying project contains the exact model and dataset revisions, frozen settings, question splits, raw responses, measured timings, search histories and all four trained neural checkpoints. Its README explains how to regenerate the analysis and report. Detailed execution logs and accounting records remain available there for reproduction; they are separate from the scientific findings presented here.

This experiment established a usable baseline and completed the planned comparison within budget. It did not show that neural-guided tuning improved on the shared starting configurations or outperformed random search. The selected serving settings had higher observed held-out accuracy, but the evidence was not strong enough to establish a repeatable improvement. Allowing longer outputs improved the number of correct answers under both configurations at the same total allocation, with no observed interaction between serving configuration and output allowance.

## References

[1] Cheng et al. (2025). [SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines](https://arxiv.org/abs/2408.04323v2).

[2] Snoek et al. (2015). [Scalable Bayesian Optimization Using Deep Neural Networks](https://proceedings.mlr.press/v37/snoek15.html). ICML, PMLR 37, 2171-2180.

[3] Qwen Team. [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B).

[4] vLLM Project. [vLLM 0.10.2 source](https://github.com/vllm-project/vllm/tree/v0.10.2).

[5] OpenAI. [GSM8K dataset and evaluation repository](https://github.com/openai/grade-school-math).
'''
    (ROOT/'REPORT.md').write_text(text.replace('–','-').replace('—','-'))
    print(ROOT/'REPORT.md')

if __name__=='__main__':main()
