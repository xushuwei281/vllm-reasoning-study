"""Explicit entry points: inspect, prepare, validate, collect, search and report."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from .data import write_json,read_jsonl


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    e=sub.add_parser('environment');e.add_argument('--output',required=True)
    p=sub.add_parser('preflight');p.add_argument('--vllm',default='vllm');p.add_argument('--version',default='0.32.0');p.add_argument('--output',required=True)
    d=sub.add_parser('prepare-data');d.add_argument('--directory',required=True);d.add_argument('--split',choices=['train','test'],default='train');d.add_argument('--freeze')
    f=sub.add_parser('fake-smoke');f.add_argument('--output',required=True)
    demo=sub.add_parser('synthetic-demo');demo.add_argument('--output',required=True);demo.add_argument('--epochs',type=int,default=100)
    rp=sub.add_parser('replay');rp.add_argument('--table',required=True);rp.add_argument('--output',required=True);rp.add_argument('--scenario',required=True);rp.add_argument('--budget',type=float,default=1000);rp.add_argument('--max-trial-cost',type=float,default=100)
    for name in ('run','search'):
        r=sub.add_parser(name)
        for flag in ('study','environment','capabilities','data','output','vllm','ledger'):r.add_argument('--'+flag,required=True)
        r.add_argument('--scenario',default='short-low');r.add_argument('--tokenizer-path');r.add_argument('--hf-home')
        r.add_argument('--seed',type=int,default=724);r.add_argument('--reference-accuracy',type=float);r.add_argument('--port',type=int,default=8137)
        if name=='run':
            r.add_argument('--config-index',type=int,default=0);r.add_argument('--default',action='store_true');r.add_argument('--run-id',required=True)
            r.add_argument('--partition',choices=['profiling','development_validation'],default='profiling')
            r.add_argument('--trace');r.add_argument('--question-count',type=int)
        else:
            r.add_argument('--method',choices=['random','gp','neural'],required=True);r.add_argument('--max-trial-cost',type=float,required=True)
    rr=sub.add_parser('reproduce-run')
    for flag in ('study','original-run','data','tokenizer-path','environment','capabilities','vllm','ledger','output','run-id'):rr.add_argument('--'+flag,required=True)
    rr.add_argument('--hf-home');rr.add_argument('--port',type=int,default=8137)
    fr=sub.add_parser('freeze');fr.add_argument('--study',required=True);fr.add_argument('--selected',required=True);fr.add_argument('--train-manifest',required=True);fr.add_argument('--output',required=True)
    fc=sub.add_parser('fixed-compute')
    for flag in ('freeze','data','environment','capabilities','vllm','output','tokenizer-path','ledger'):fc.add_argument('--'+flag,required=True)
    fc.add_argument('--hf-home');fc.add_argument('--seed',type=int,default=724)
    pl=sub.add_parser('plot');pl.add_argument('--measured-dir',required=True)
    args=parser.parse_args(argv)
    if args.command=='environment':
        from .environment import collect_environment,local_capabilities
        value=collect_environment();write_json(args.output,value);write_json(str(Path(args.output).with_name('capabilities.json')),local_capabilities(value));print(value['environment_hash'])
    elif args.command=='preflight':
        from .config import preflight_runtime
        value=preflight_runtime(args.version,args.vllm,args.output);write_json(Path(args.output)/'preflight.json',value);print(json.dumps({'ok':value['ok'],'errors':value['errors']}));return 0 if value['ok'] else 2
    elif args.command=='prepare-data':
        from .data import prepare_dataset
        value=prepare_dataset(args.directory,args.split,freeze_path=args.freeze);print(json.dumps({k:v for k,v in value.items() if k!='partitions'}))
    elif args.command=='fake-smoke':
        from .fake_server import fixture_run
        print(json.dumps(asyncio.run(fixture_run(args.output)),indent=2))
    elif args.command=='synthetic-demo':
        from .replay import synthetic_table,replay,train_diagnostic
        from .data import write_jsonl
        from .plots import replay_figures,prediction_figure
        p=Path(args.output);p.mkdir(parents=True,exist_ok=True);table=synthetic_table();write_jsonl(p/'table.jsonl',table)
        replay(table,p/'replay',neural_epochs=args.epochs);train_diagnostic(table,p/'model',epochs=args.epochs)
        replay_figures(p/'replay');prediction_figure(p/'model');print('Synthetic software validation complete; not research results.')
    elif args.command=='replay':
        from .replay import replay
        from .plots import replay_figures
        table=read_jsonl(args.table);replay(table,args.output,scenario=args.scenario,budget_s=args.budget,maximum_trial_cost_s=args.max_trial_cost)
        replay_figures(args.output,source=table[0]['source'])
    elif args.command in ('run','search'):
        from transformers import AutoTokenizer
        from .experiment import prepare_workload,candidates_for,run_trial,closed_loop,trial_reservation_bound
        from .data import digest
        study=json.loads(Path(args.study).read_text());env=json.loads(Path(args.environment).read_text());caps=json.loads(Path(args.capabilities).read_text())
        workload=next(w for w in study['workloads'] if w['scenario_id']==args.scenario)
        if workload.get('partition')=='holdout' and args.command=='search':raise ValueError('cannot select using held-out workload')
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path or study['model']['name'],revision=study['model']['tokenizer_revision'])
        qs,ps,desc=prepare_workload(study,workload,args.data,tokenizer,partition=getattr(args,'partition','profiling'),limit=getattr(args,'question_count',None))
        candidates=candidates_for(study,desc);runtime_env=dict(os.environ);runtime_env['TOKENIZERS_PARALLELISM']='false'
        if args.hf_home:runtime_env['HF_HOME']=str(Path(args.hf_home).resolve())
        common=dict(study=study,capabilities=caps,environment=env,questions=qs,prompts=ps,executable=args.vllm,
                    tokenizer=tokenizer,seed=args.seed,reference_accuracy=args.reference_accuracy,runtime_env=runtime_env,port=args.port,ledger_path=args.ledger)
        if args.command=='run':
            candidate=candidates[args.config_index]
            if args.default:
                candidate['config'].update(max_num_batched_tokens=None,max_num_seqs=None)
                candidate['config_id']=digest(candidate['config'])[:12];candidate['is_default']=True
            trace=json.loads(Path(args.trace).read_text()) if args.trace else None
            row=run_trial(candidate=candidate,output_root=args.output,run_id=args.run_id,trace=trace,**common)
            print(json.dumps(row,indent=2))
        else:
            from .config import validate_config
            candidates=[c for c in candidates if not validate_config(c['config'],caps,c['workload'])]
            def runner(c,index):
                return run_trial(candidate=c,output_root=Path(args.output)/'live',run_id=f'{args.method}-seed{args.seed}-{index:03d}',**{**common,'seed':args.seed+index})
            bound=trial_reservation_bound(study)['accelerator_seconds']
            if args.max_trial_cost<bound:raise ValueError(f'--max-trial-cost must cover full reservation bound {bound}')
            rows=closed_loop(candidates,runner,Path(args.output)/'search',method=args.method,seed=args.seed,
                             initial_points=study['budget']['initial_points'],accelerator_budget_s=study['budget']['search_method_seconds'],maximum_trial_cost_s=args.max_trial_cost)
            print(f'{len(rows)} observed trials; independent validation required')
    elif args.command=='reproduce-run':
        from transformers import AutoTokenizer
        from .data import select_questions,prompt_for,digest
        from .experiment import run_trial
        origin=Path(args.original_run)
        original=json.loads((origin/'observation.json').read_text());plan=json.loads((origin/'plan.json').read_text())
        provenance=json.loads((origin/'provenance.json').read_text());study=json.loads(Path(args.study).read_text())
        if digest(study)!=provenance['study_hash']:raise ValueError('study does not match original run')
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path)
        lookup={q['question_id']:q for q in select_questions(args.data,'profiling')}
        questions=[lookup[q] for q in plan['question_ids']]
        candidate=provenance['candidate']
        controls=provenance.get('execution_controls',{})
        if original['source']!='measured':raise ValueError('reproduce-run accepts real serving records only')
        prompts=[prompt_for(q['question'],tokenizer,candidate['workload']['profile']) for q in questions]
        if [digest(p) for p in prompts]!=plan['prompt_hashes']:raise ValueError('tokenizer/template mismatch')
        runtime_env=dict(os.environ);runtime_env['TOKENIZERS_PARALLELISM']='false'
        if args.hf_home:runtime_env['HF_HOME']=str(Path(args.hf_home).resolve())
        row=run_trial(candidate=candidate,study=study,capabilities=json.loads(Path(args.capabilities).read_text()),
           environment=json.loads(Path(args.environment).read_text()),questions=questions,prompts=prompts,
           output_root=args.output,executable=args.vllm,tokenizer=tokenizer,seed=original['seed'],runtime_env=runtime_env,
           run_id=args.run_id,trace=plan['trace'],protocol_override=plan['protocol'],ledger_path=args.ledger,port=args.port,
           repetition=controls.get('repetition',original.get('repetition',0)),
           reference_accuracy=controls.get('reference_accuracy'),deadline_mode=controls.get('deadline_mode',False),
           phase=controls.get('phase','calibration'))
        print(json.dumps(row,indent=2))
    elif args.command=='freeze':
        from .data import freeze_protocol
        freeze_protocol(args.output,json.loads(Path(args.study).read_text()),json.loads(Path(args.selected).read_text()),json.loads(Path(args.train_manifest).read_text()))
    elif args.command=='fixed-compute':
        from transformers import AutoTokenizer
        from .fixed_compute import run_fixed_compute
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_path)
        runtime_env=dict(os.environ)
        if args.hf_home:runtime_env['HF_HOME']=str(Path(args.hf_home).resolve())
        run_fixed_compute(args.freeze,data_dir=args.data,tokenizer=tokenizer,capabilities=json.loads(Path(args.capabilities).read_text()),environment=json.loads(Path(args.environment).read_text()),executable=args.vllm,output_root=args.output,runtime_env=runtime_env,seed=args.seed,ledger_path=args.ledger)
    elif args.command=='plot':
        from .plots import measured_figures
        measured_figures(args.measured_dir)
    return 0

if __name__=='__main__': raise SystemExit(main())
