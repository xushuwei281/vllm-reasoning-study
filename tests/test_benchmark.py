import asyncio
from vllm_tuner.benchmark import arrival_trace, collect
from vllm_tuner.fake_server import make_app
from aiohttp import web

def test_arrivals_independent_of_service_and_reproducible():
    a=arrival_trace(rate=10,warmup_s=1,ramp_s=1,measurement_s=3,seed=1)
    assert a==arrival_trace(rate=10,warmup_s=1,ramp_s=1,measurement_s=3,seed=1)
    assert any(x['phase']=='measure' for x in a)
    assert all(0<x['intended_arrival_s']<5 for x in a)

def test_cap_rejects_instead_of_silently_throttling_and_times_out(tmp_path):
    async def run():
        runner=web.AppRunner(await make_app(delay=.2));await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        protocol=dict(arrival_rate=100,warmup_s=0,ramp_s=0,measurement_s=.1,drain_s=0,timeout_s=.05,max_in_flight=1,telemetry_interval_s=.1)
        trace=[{'intended_arrival_s':i*.001,'phase':'measure'} for i in range(20)]
        try:
            rows,_,info=await collect(f'http://127.0.0.1:{port}','fake',[{'question_id':'q','gold':'4'}],['2+2'],{},protocol,tmp_path,source='synthetic',trace=trace)
            assert len(rows)==20 and info['offered']==20
            assert any(r['status']=='rejected' for r in rows)
            assert any(r['status']=='timeout' for r in rows)
            assert all(r['correct'] is False for r in rows)
        finally: await runner.cleanup()
    asyncio.run(run())
