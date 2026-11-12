"""Staged continuation on one private, timeout-bounded Hugging Face GPU job.

Run from the project directory. Test data is only opened by the final stage,
after measured independent validation and the immutable freeze validator.
"""
from pathlib import Path
import argparse, copy, json, os, shutil, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vllm_tuner.data import digest, write_json, read_jsonl, select_questions, prompt_for
from vllm_tuner.experiment import run_trial, candidates_for, closed_loop

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'continuation'
LEDGER=ROOT/'results'/'measured'/'budget.jsonl'
DATA=ROOT/'data'

def reconcile_allocated_overhead(label,phase='optimizer'):
    """Charge all elapsed cloud allocation once, including CPU/idle intervals."""
    from datetime import datetime
    from vllm_tuner.accounting import BudgetLedger
    allocation=json.loads((OUT/'cloud-allocation.json').read_text())
    created=datetime.fromisoformat(allocation['created_at']).timestamp()
    ledger=BudgetLedger(LEDGER,28800,.25)
    recorded=sum(r['accelerator_seconds'] for r in read_jsonl(LEDGER) if r['event']=='finish')
    cloud_recorded=recorded-allocation['preallocation_charged_accelerator_seconds']
    residual=max(0,time.time()-created-cloud_recorded)
    run_id='cloud-overhead-'+label
    if any(r.get('run_id')==run_id for r in read_jsonl(LEDGER)):
        raise RuntimeError('overhead reconciliation label must be unique')
    ledger.begin(run_id,1,residual,phase)
    ledger.finish(run_id,residual,status='allocated_overhead')
    print('ALLOCATION_OVERHEAD',label,residual,flush=True)

def setup():
    from vllm_tuner.environment import collect_environment, local_capabilities
    from vllm_tuner.config import preflight_runtime
    from huggingface_hub import snapshot_download
    OUT.mkdir(parents=True,exist_ok=True)
    env=collect_environment()
    env['container_image']='vllm/vllm-openai:v0.10.2'
    from vllm_tuner.environment import environment_hash
    env['environment_hash']=environment_hash(env)
    caps=local_capabilities(env); caps['max_model_len']=8192
    if (OUT/'environment.json').exists():
        saved=json.loads((OUT/'environment.json').read_text())
        if saved['environment_hash']!=env['environment_hash']:raise RuntimeError('setup environment changed after inventory')
    else:write_json(OUT/'environment.json',env,immutable=True)
    write_json(OUT/'capabilities.json',caps,immutable=True)
    preflight=preflight_runtime('0.10.2',shutil.which('vllm'),str(OUT/'preflight'))
    if (OUT/'preflight.json').exists():
        old=json.loads((OUT/'preflight.json').read_text())
        if not old['ok']:
            (OUT/'preflight.json').rename(OUT/('preflight-failed-'+digest(old)[:12]+'.json'))
    write_json(OUT/'preflight.json',preflight,immutable=True)
    print('PREFLIGHT',json.dumps({k:v for k,v in preflight.items() if k.endswith('ok') or k=='errors'}),flush=True)
    if not preflight['ok']: raise RuntimeError('runtime preflight failed')
    base=json.loads((ROOT/'configs/study.json').read_text())
    snapshot_download(base['model']['name'],revision=base['model']['revision'])
    print('MODEL_READY',flush=True)

def context():
    from transformers import AutoTokenizer
    study=json.loads((ROOT/'configs/study-cloud.json').read_text())
    env=json.loads((OUT/'environment.json').read_text())
    caps=json.loads((OUT/'capabilities.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(study['model']['name'],revision=study['model']['tokenizer_revision'],local_files_only=True)
    return study,env,caps,tokenizer

def trial(study,env,caps,tok,config,workload,questions,run_id,*,default=False,phase='calibration',protocol=None,trace=None,reference=None,seed=724):
    prompts=[prompt_for(q['question'],tok) for q in questions]
    lengths=[len(tok.encode(p,add_special_tokens=False)) for p in prompts]
    workload={**workload,'input_tokens_max':max(lengths),'input_tokens_mean':sum(lengths)/len(lengths),'input_tokens_p95':float(sorted(lengths)[min(len(lengths)-1,int(.95*len(lengths)))])}
    candidate={'config_id':digest(config)[:12],'config':config,'scenario_id':workload['scenario_id'],'workload':workload,'known_valid':True,'is_default':default}
    row=run_trial(candidate=candidate,study=study,capabilities=caps,environment=env,questions=questions,prompts=prompts,
        output_root=OUT,executable=shutil.which('vllm'),tokenizer=tok,seed=seed,reference_accuracy=reference,phase=phase,
        run_id=run_id,protocol_override=protocol,trace=trace,ledger_path=LEDGER,port=8137)
    print('TRIAL',json.dumps({'run_id':run_id,'status':row['status'],'metrics':row['metrics'],'cost':row['cost']}),flush=True)
    return row

def pilot(spec_path):
    s,e,c,t=context(); spec=json.loads(Path(spec_path).read_text())
    q=select_questions(DATA,spec.get('partition','calibration'))[:spec['question_count']]
    config={**s['fixed_config'],**spec.get('config',{})}
    workload={**s['workloads'][0],**spec.get('workload',{})}
    if 'decoding' in spec:s['decoding'].update(spec['decoding'])
    return trial(s,e,c,t,config,workload,q,spec['run_id'],default=spec.get('default',False),protocol=spec.get('protocol'),trace=spec.get('trace'),reference=spec.get('reference_accuracy'))

def search(method,max_trials=None):
    s,e,c,t=context()
    plan=json.loads((OUT/'search-plan.json').read_text())
    questions=select_questions(DATA,'profiling')[:plan['question_count']]
    workload=s['workloads'][0]
    prompts=[prompt_for(q['question'],t) for q in questions]
    lengths=[len(t.encode(p,add_special_tokens=False)) for p in prompts]
    workload={**workload,'input_tokens_mean':sum(lengths)/len(lengths),'input_tokens_p95':float(sorted(lengths)[int(.95*len(lengths))]),'input_tokens_max':max(lengths)}
    candidates=candidates_for(s,workload)
    def runner(candidate,index):
        row=run_trial(candidate=candidate,study=s,capabilities=c,environment=e,questions=questions,prompts=prompts,
            output_root=OUT,executable=shutil.which('vllm'),tokenizer=t,seed=plan['trace_seed'],reference_accuracy=plan['reference_accuracy'],
            phase='search',run_id=f'live-{method}-{index:02d}',protocol_override=plan['protocol'],trace=plan['traces'][index],ledger_path=LEDGER,port=8137)
        print('TRIAL',json.dumps({'method':method,'index':index,'run_id':row['run_id'],'status':row['status'],'metrics':row['metrics'],'cost':row['cost']}),flush=True)
        return row
    rows=closed_loop(candidates,runner,OUT/'search'/method,method=method,seed=plan['search_seed'],initial_points=6,
        accelerator_budget_s=plan['method_budget_s'],maximum_trial_cost_s=plan['trial_bound_s'],neural_epochs=200,max_trials=max_trials)
    print('SEARCH_FINISHED',method,len(rows),flush=True)

def campaign():
    plan=json.loads((OUT/'search-plan.json').read_text())
    from vllm_tuner.live_analysis import validate_planned_trace_hashes
    validate_planned_trace_hashes([digest(t) for t in plan['traces']],required_count=plan['trials_per_method'])
    # Each method has its own paid observations and policy state. Rotate order
    # at each round to distribute temporal hardware drift across methods.
    methods=['random','gp','neural']
    reconcile_allocated_overhead('before-main','calibration')
    for index in range(plan['trials_per_method']):
        order=methods[index%3:]+methods[:index%3]
        for method in order:
            search(method,index+1)
            reconcile_allocated_overhead(f'round{index:02d}-{method}')
    from vllm_tuner.live_analysis import summarize_live_search
    rows={m:read_jsonl(OUT/'search'/m/'observations.jsonl') for m in methods}
    write_json(OUT/'live-search-summary.json',summarize_live_search(rows,common_initial_count=6,expected_trace_hashes=[digest(trace) for trace in plan['traces']]),immutable=True)
    print('CAMPAIGN_COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['setup','pilot','search','campaign']); p.add_argument('argument',nargs='?'); a=p.parse_args()
    if a.stage=='setup':setup()
    elif a.stage=='pilot':pilot(a.argument)
    elif a.stage=='search':search(a.argument)
    elif a.stage=='campaign':campaign()

if __name__=='__main__':main()
