"""Synthetic HTTP fixture. Never produces research measurements."""
import asyncio
import json
from aiohttp import web

async def make_app(delay=0.004):
    app=web.Application(); state={'active':0}
    async def health(request): return web.json_response({'status':'synthetic'})
    async def metrics(request):
        return web.Response(text=f'vllm:num_requests_running {state["active"]}\nvllm:num_requests_waiting 0\n')
    async def completion(request):
        payload=await request.json(); state['active']+=1
        try:
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
            await response.prepare(request)
            text='Synthetic thinking. </think> The answer is \\boxed{4}.'
            pieces=['Synthetic thinking. ','</think> ','The answer is ','\\boxed{4}.']
            for piece in pieces:
                await asyncio.sleep(delay)
                await response.write(('data: '+json.dumps({'choices':[{'text':piece,'finish_reason':None}]})+'\n\n').encode())
            await response.write(('data: '+json.dumps({'choices':[{'text':'','finish_reason':'stop'}], 'usage':{'completion_tokens':12,'prompt_tokens':10}})+'\n\ndata: [DONE]\n\n').encode())
            await response.write_eof(); return response
        finally: state['active']-=1
    app.router.add_get('/health',health);app.router.add_get('/metrics',metrics)
    app.router.add_post('/v1/completions',completion)
    return app

async def fixture_run(directory):
    from .benchmark import collect, evaluate_collected
    from .evaluation import summarize_requests
    from .data import write_json,write_jsonl
    app=await make_app(); runner=web.AppRunner(app); await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    questions=[{'question_id':'synthetic:2plus2','question':'2+2?','gold':'#### 4'}]
    protocol=dict(arrival_rate=40,warmup_s=.1,ramp_s=.1,measurement_s=1,drain_s=1,timeout_s=1,max_in_flight=50,telemetry_interval_s=.1)
    try:
        requests,samples,info=await collect(f'http://127.0.0.1:{port}','synthetic-fixture',questions,['2+2?'],{'max_tokens':32,'seed':1},protocol,directory,source='synthetic',seed=42)
        scored=evaluate_collected(requests,questions)
        write_jsonl(str(directory)+'/scored.jsonl',scored)
        summary=summarize_requests(scored,window_start_s=info['window_start_s'],window_end_s=info['window_end_s'],slos={'ttft_s':.5,'tpot_s':.5,'e2e_s':1},allocated_accelerators=0)
        write_json(str(directory)+'/summary.json',{'source':'synthetic','metrics':summary},immutable=True)
        return summary
    finally: await runner.cleanup()
