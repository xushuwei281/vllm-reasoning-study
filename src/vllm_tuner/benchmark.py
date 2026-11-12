"""Open-loop streaming workload; no semaphore that silently changes offered load."""
from __future__ import annotations
import asyncio
import json
import math
from pathlib import Path
import random
import shutil
import time
import httpx
from .data import digest, write_json, write_jsonl


def arrival_trace(*, rate, warmup_s, ramp_s, measurement_s, seed=0):
    if rate <= 0 or measurement_s <= 0 or min(warmup_s,ramp_s) < 0:
        raise ValueError("positive rate/duration required")
    rng=random.Random(seed); rows=[]
    segments=[(0.0,warmup_s,rate*.25,'warmup')]
    for i in range(4):
        segments.append((warmup_s+i*ramp_s/4,warmup_s+(i+1)*ramp_s/4,rate*(i+1)/4,'ramp'))
    segments.append((warmup_s+ramp_s,warmup_s+ramp_s+measurement_s,rate,'measure'))
    for start,end,intensity,phase in segments:
        t=start
        while intensity>0:
            t+=rng.expovariate(intensity)
            if t>=end: break
            rows.append({'intended_arrival_s':t,'phase':phase})
    return rows


def validate_trace(trace,protocol):
    """Reject unbounded schedules before any reservation or output creation."""
    for key in ('measurement_s','timeout_s','max_in_flight'):
        if not math.isfinite(protocol[key]) or protocol[key]<=0:raise ValueError(f'invalid {key}')
    for key in ('warmup_s','ramp_s','drain_s'):
        if not math.isfinite(protocol[key]) or protocol[key]<0:raise ValueError(f'invalid {key}')
    end=protocol['warmup_s']+protocol['ramp_s']+protocol['measurement_s']
    previous=-1.
    for item in trace:
        t=item['intended_arrival_s']
        if not math.isfinite(t) or t<previous or not 0<=t<end:raise ValueError('arrival trace must be sorted and within offered window')
        expected='warmup' if t<protocol['warmup_s'] else ('ramp' if t<protocol['warmup_s']+protocol['ramp_s'] else 'measure')
        if item['phase']!=expected:raise ValueError('trace phase does not match frozen boundaries')
        previous=t


def _empty(request_id, question_id, item):
    return dict(request_id=request_id,question_id=question_id,phase=item['phase'],
                intended_arrival_s=item['intended_arrival_s'],dispatch_s=None,
                first_token_s=None,completion_s=None,status='pending',text='',
                output_tokens=None,input_tokens=None,finish_reason=None,correct=False,chunks=[])


async def collect(base_url, model, questions, prompts, decoding, protocol, output_dir,
                  *, source='measured', seed=0, deadline_mode=False, trace=None):
    """Save every offered request, including client rejections and cancellations.

    `prompts` are already templated strings; questions carry evaluator-only gold.
    No correctness or tokenization work runs on the timing-critical streaming path.
    In deadline mode the fixed cohort is dispatched open-loop by time zero/default
    trace and all outstanding work is cancelled at the shared measured deadline.
    """
    if source not in {'measured','synthetic'}: raise ValueError('explicit source required')
    if not questions or len(questions)!=len(prompts): raise ValueError('question/prompt mismatch')
    validate_trace(trace or [],protocol)
    p=Path(output_dir); p.mkdir(parents=True,exist_ok=True)
    if (p/'requests.jsonl').exists() or (p/'offered.jsonl').exists():
        raise FileExistsError('raw run directory is immutable; use a new run ID for retries')
    warm=protocol['warmup_s']; ramp=protocol['ramp_s']; duration=protocol['measurement_s']
    end=warm+ramp+duration
    trace=trace if trace is not None else arrival_trace(rate=protocol['arrival_rate'],warmup_s=warm,ramp_s=ramp,measurement_s=duration,seed=seed)
    validate_trace(trace,protocol)
    if deadline_mode and len(trace)!=len(questions): raise ValueError('fixed cohort requires one arrival per question')
    write_json(p/'plan.json',{'source':source,'trace':trace,'protocol':protocol,'decoding':decoding,
                             'question_ids':[q['question_id'] for q in questions],
                             'prompt_hashes':[digest(x) for x in prompts]},immutable=True)
    # Proxy/environment settings are excluded to avoid routing localhost through corporate proxies.
    client=httpx.AsyncClient(timeout=httpx.Timeout(protocol['timeout_s']),trust_env=False,
                             limits=httpx.Limits(max_connections=protocol['max_in_flight']+4,max_keepalive_connections=protocol['max_in_flight']+4))
    epoch=time.perf_counter(); now=lambda: time.perf_counter()-epoch
    active=set(); results=[]; telemetry=[]; stop_monitor=asyncio.Event()
    offered_file=(p/'offered.jsonl').open('x'); result_file=(p/'requests.jsonl').open('x')
    metrics_file=(p/'server-metrics.jsonl').open('x')
    def commit(row):
        row['source']=source; results.append(row)
        result_file.write(json.dumps(row,allow_nan=False)+'\n'); result_file.flush()
    async def request(index,item):
        q=questions[index%len(questions)]; row=_empty(f'r{index:06d}',q['question_id'],item)
        row['dispatch_s']=now(); usage=None
        payload={'model':model,'prompt':prompts[index%len(prompts)],'stream':True,
                 'stream_options':{'include_usage':True},**decoding}
        # The same question/repetition uses the same random seed across configurations.
        payload['seed']=int(decoding.get('seed',0))+index
        row['sampling_seed']=payload['seed']
        if item['phase']!='measure':
            payload['max_tokens']=protocol.get('warmup_output_token_limit',payload.get('max_tokens'))
        row['requested_output_token_limit']=payload.get('max_tokens')
        try:
            async with asyncio.timeout(protocol['timeout_s']), client.stream('POST',base_url.rstrip('/')+'/v1/completions',json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data:'): continue
                    raw=line[5:].strip()
                    if raw=='[DONE]': break
                    event=json.loads(raw)
                    if event.get('error'): raise RuntimeError(str(event['error']))
                    if event.get('usage'):
                        usage=event['usage']
                        row['output_tokens']=usage.get('completion_tokens'); row['input_tokens']=usage.get('prompt_tokens')
                    for choice in event.get('choices',[]):
                        piece=choice.get('text','')
                        if piece:
                            stamp=now()
                            if row['first_token_s'] is None: row['first_token_s']=stamp
                            row['text']+=piece
                            row['chunks'].append({'time_s':stamp,'token_count':None})
                        if choice.get('finish_reason'): row['finish_reason']=choice['finish_reason']
            if not usage or row['finish_reason'] is None:
                raise RuntimeError('incomplete stream or missing usage; token counts cannot be inferred as server counts')
            row['output_tokens']=int(usage['completion_tokens']); row['input_tokens']=int(usage['prompt_tokens'])
            row['status']='ok'; row['completion_s']=now()
        except (httpx.TimeoutException,asyncio.CancelledError,TimeoutError):
            row['status']='timeout'; row['completion_s']=now()
        except Exception as exc:
            row['status']='error'; row['completion_s']=now(); row['error']=f'{type(exc).__name__}: {exc}'
        finally:
            commit(row)
    async def monitor():
        has_nvidia=bool(shutil.which('nvidia-smi')) and bool(protocol.get('allocated_gpu_ids'))
        while not stop_monitor.is_set():
            sample={'time_s':now(),'client_active_requests':len(active),'device_memory_bytes':None}
            try:
                res=await client.get(base_url.rstrip('/')+'/metrics',timeout=2)
                if res.status_code==200:
                    metrics_file.write(json.dumps({'time_s':now(),'prometheus':res.text})+'\n'); metrics_file.flush()
                    for line in res.text.splitlines():
                        for key in ('vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc','vllm:gpu_cache_usage_perc'):
                            if line.startswith(key+'{') or line.startswith(key+' '):
                                sample[key]=sample.get(key,0)+float(line.rsplit(' ',1)[-1])
                else: sample['metrics_http_status']=res.status_code
            except Exception as exc: sample['telemetry_error']=type(exc).__name__
            if has_nvidia:
                try:
                    proc=await asyncio.create_subprocess_exec('nvidia-smi','--id='+','.join(str(x) for x in protocol['allocated_gpu_ids']),'--query-gpu=memory.used','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
                    out,_=await asyncio.wait_for(proc.communicate(),2)
                    # Whole visible fleet; never substitute host RSS for accelerator allocation.
                    sample['device_memory_bytes']=sum(float(x) for x in out.decode().splitlines())*1024**2
                except Exception as exc:
                    sample['memory_error']=type(exc).__name__
                    if 'proc' in locals() and proc.returncode is None:
                        proc.kill(); await proc.wait()
            telemetry.append(sample)
            try: await asyncio.wait_for(stop_monitor.wait(),protocol.get('telemetry_interval_s',1))
            except asyncio.TimeoutError: pass
    monitor_task=asyncio.create_task(monitor())
    try:
        for index,item in enumerate(trace):
            await asyncio.sleep(max(0,item['intended_arrival_s']-now()))
            offered={'request_id':f'r{index:06d}','question_id':questions[index%len(questions)]['question_id'],**item}
            offered_file.write(json.dumps(offered)+'\n'); offered_file.flush()
            if len(active)>=protocol['max_in_flight']:
                row=_empty(offered['request_id'],offered['question_id'],item)
                row.update(status='rejected',completion_s=now(),error='client_in_flight_cap')
                commit(row); continue
            task=asyncio.create_task(request(index,item)); active.add(task); task.add_done_callback(active.discard)
        await asyncio.sleep(max(0,end-now()))
        if active:
            _,pending=await asyncio.wait(active,timeout=0 if deadline_mode else protocol['drain_s'])
            for task in pending: task.cancel()
            await asyncio.gather(*pending,return_exceptions=True)
    finally:
        for task in active.copy(): task.cancel()
        await asyncio.gather(*active,return_exceptions=True)
        stop_monitor.set(); await monitor_task
        await client.aclose(); offered_file.close(); result_file.close(); metrics_file.close()
    write_jsonl(p/'telemetry.jsonl',telemetry)
    info={'source':source,'window_start_s':warm+ramp,'window_end_s':end,'actual_wall_seconds':now(),
          'offered':len(trace),'recorded':len(results),'trace_hash':digest(trace),
          'ttft_observation':'first nonempty text-bearing SSE chunk; transport/chunk coalescing may delay observation',
          'inter_token_latency':'unavailable within multi-token chunks; chunk gaps saved separately'}
    write_json(p/'collection.json',info,immutable=True)
    return sorted(results,key=lambda x:x['request_id']),telemetry,info


def evaluate_collected(requests,questions,tokenizer=None):
    from .evaluation import score_answer
    gold={x['question_id']:x['gold'] for x in questions}; rows=[]
    for original in requests:
        row=dict(original)
        row['score']=score_answer(row['text'],gold[row['question_id']],status=row['status'],finish_reason=row['finish_reason'],require_thinking_close=True)
        row['correct']=row['score']['correct']
        if tokenizer is not None:
            text=row['text']; marker='</think>'
            reasoning,final=text.split(marker,1) if marker in text else (text,'')
            row['reasoning_tokens_text_reencoded']=len(tokenizer.encode(reasoning,add_special_tokens=False))
            row['final_tokens_text_reencoded']=len(tokenizer.encode(final,add_special_tokens=False))
            row['token_partition_note']='re-encoded text counts; server usage is authoritative for total tokens'
        rows.append(row)
    return rows
