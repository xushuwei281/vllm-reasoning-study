"""CPU-only reservation timing checks; fake clocks never launch a server."""
import copy
import json
import pytest

from vllm_tuner import experiment
from vllm_tuner.config import BASE_CONFIG, MODEL


@pytest.fixture
def slot_trial(tmp_path, monkeypatch):
    class Clock:
        now = 100.
        sleep_overshoot = 0.
        def monotonic(self):
            return self.now
        def sleep(self, seconds):
            assert seconds >= 0
            self.now += seconds + self.sleep_overshoot
    clock = Clock()

    class Server:
        starts = 0
        requested_startup = 10.
        last_timeout = None
        def __init__(self, command, path, devices, startup, shutdown, **kwargs):
            self.started_at = self.stopped_at = None
            self.startup_timeout = startup
            Server.last_timeout = startup
        def start(self):
            Server.starts += 1
            self.started_at = clock.now
            clock.now += min(self.requested_startup, self.startup_timeout)
            if self.requested_startup > self.startup_timeout:
                raise TimeoutError('fake startup exhausted allocation')
        def stop(self):
            if self.started_at is not None and self.stopped_at is None:
                clock.now += 4.
                self.stopped_at = clock.now
        def cost(self):
            wall = 0. if self.started_at is None else self.stopped_at - self.started_at
            return {'wall_seconds':wall, 'accelerator_seconds':wall, 'allocated_devices':1,
                    'optimizer_cpu_seconds':0.}

    collection = {'extra_seconds':0., 'protocol':None}
    async def collect(*args, **kwargs):
        protocol = args[5]
        collection['protocol'] = copy.deepcopy(protocol)
        clock.now += protocol['warmup_s'] + protocol['ramp_s'] + protocol['measurement_s'] + collection['extra_seconds']
        row = {'request_id':'r0', 'question_id':'q0', 'phase':'measure',
               'intended_arrival_s':5., 'dispatch_s':5., 'first_token_s':5.1,
               'completion_s':5.2, 'status':'ok', 'text':'<think>one</think>\\boxed{1}',
               'input_tokens':3, 'output_tokens':4, 'finish_reason':'stop', 'chunks':[]}
        return [row], [], {'window_start_s':5., 'window_end_s':5.+protocol['measurement_s'], 'trace_hash':'fake'}
    original_evaluate = experiment.evaluate_collected
    def evaluate(*args):
        clock.now += 2.
        return original_evaluate(*args)
    monkeypatch.setattr(experiment, 'ServerProcess', Server)
    monkeypatch.setattr(experiment, 'collect', collect)
    monkeypatch.setattr(experiment, 'evaluate_collected', evaluate)
    monkeypatch.setattr(experiment.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(experiment.time, 'sleep', clock.sleep)
    study = {'model':copy.deepcopy(MODEL), 'startup_timeout_s':200,
             'budget':{'total_accelerator_seconds':1000, 'validation_fraction':.25, 'allocated_devices':1},
             'benchmark':{'warmup_s':2., 'ramp_s':3., 'measurement_s':20., 'drain_s':0.,
                          'timeout_s':20., 'max_in_flight':4, 'reserved_wall_seconds':60.,
                          'teardown_allowance_s':30., 'deadline_includes_startup':True},
             'decoding':{'temperature':.6, 'seed':0},
             'slos':{'ttft_s':1., 'tpot_s':1., 'e2e_s':1.},
             'constraints':{'max_accuracy_drop':.02, 'max_error_rate':.01}}
    values = dict(study=study, candidate={'config_id':'unit', 'scenario_id':'slot',
        'config':copy.deepcopy(BASE_CONFIG), 'workload':{'arrival_rate':1., 'output_token_limit':1024,
        'input_tokens_max':3, 'profile':'short'}},
        capabilities={'device_count':1, 'allowed_tp_sizes':[1], 'supported_precisions':['bf16'],
                      'chunked_prefill':True, 'prefix_caching':True, 'max_model_len':4096, 'max_memory_fraction':.65},
        environment={'environment_hash':'unit'}, questions=[{'question_id':'q0', 'question':'one?', 'gold':'#### 1'}],
        prompts=['unit'], output_root=tmp_path/'out', executable='/fake/vllm', run_id='slot-1',
        reference_accuracy=1., runtime_env={'UNIT':'1'}, ledger_path=tmp_path/'ledger.jsonl',
        deadline_mode=True, trace=[{'phase':'measure', 'intended_arrival_s':5.}])
    return values, clock, Server, collection


def test_whole_slot_counts_startup_warmup_ramp_teardown_and_padding(slot_trial):
    values, clock, server, collection = slot_trial
    result = experiment.run_trial(**values)
    assert result['status'] == 'ok'
    assert server.last_timeout == 25.
    assert collection['protocol']['measurement_s'] == 15.
    assert collection['protocol']['timeout_s'] == 15.
    cost = result['cost']
    assert cost['server_lifecycle_seconds'] == 34.
    assert cost['allocated_idle_padding_seconds'] == 24.
    assert cost['allocated_after_shutdown_seconds'] == 26.
    assert cost['wall_seconds'] == cost['accelerator_seconds'] == 60.
    assert cost['admission_wall_seconds'] == 75.
    assert cost['slot_overshoot_seconds'] == 0.


def test_startup_timeout_is_capped_and_failure_still_charges_reserved_slot(slot_trial):
    values, clock, server, collection = slot_trial
    server.requested_startup = 200.
    result = experiment.run_trial(**values)
    assert result['status'] == 'failed'
    assert server.last_timeout == 25.
    assert collection['protocol'] is None
    assert result['cost']['wall_seconds'] == 60.
    assert result['cost']['server_lifecycle_seconds'] == 29.


def test_actual_overrun_is_preserved_in_cost_and_ledger(slot_trial):
    values, clock, server, collection = slot_trial
    collection['extra_seconds'] = 80.
    result = experiment.run_trial(**values)
    assert result['cost']['wall_seconds'] == 116.
    assert result['cost']['slot_overshoot_seconds'] == 56.
    assert result['cost']['admission_bound_overshoot_seconds'] == 41.
    assert result['cost']['allocated_idle_padding_seconds'] == 0.
    finish = json.loads(values['ledger_path'].read_text().splitlines()[-1])
    assert finish['accelerator_seconds'] == 116.
    assert finish['reservation_overrun'] is True


def test_scheduler_padding_overshoot_is_charged(slot_trial):
    values, clock, server, collection = slot_trial
    clock.sleep_overshoot = .001
    result = experiment.run_trial(**values)
    assert 60. < result['cost']['wall_seconds'] < 60.01
    assert result['cost']['slot_overshoot_seconds'] > 0
    assert result['cost']['admission_bound_overshoot_seconds'] == 0


@pytest.mark.parametrize('override', [
    {'reserved_wall_seconds':True}, {'reserved_wall_seconds':float('inf')},
    {'reserved_wall_seconds':float('nan')}, {'reserved_wall_seconds':-1},
    {'reserved_wall_seconds':35}, {'reserved_wall_seconds':None},
    {'teardown_allowance_s':29}, {'teardown_allowance_s':float('inf')},
    {'reservation_overhead_allowance_s':-1}, {'deadline_includes_startup':'yes'},
])
def test_invalid_reserved_protocol_never_reserves_or_launches(slot_trial, override):
    values, clock, server, collection = slot_trial
    with pytest.raises(ValueError):
        experiment.run_trial(**values, protocol_override=override)
    assert server.starts == 0
    assert not values['ledger_path'].exists()
    assert not values['output_root'].exists()


def test_whole_deadline_requires_fixed_cohort_before_reservation(slot_trial):
    values, clock, server, collection = slot_trial
    values['deadline_mode'] = False
    with pytest.raises(ValueError, match='fixed-cohort'):
        experiment.run_trial(**values)
    assert server.starts == 0
    assert not values['ledger_path'].exists()


def test_padding_slot_is_not_misrepresented_as_hard_upper_bound(slot_trial):
    values, *_ = slot_trial
    bound = experiment.trial_reservation_bound(values['study'], {'deadline_includes_startup':False})
    # min(startup=200, room=25) + warmup/ramp=5 + measure=20 + teardown=30 + overhead=15.
    assert bound['wall_seconds'] == 95.
    assert bound['accelerator_seconds'] == 95.
    assert bound['reserved_slot_seconds'] == 60.
    assert bound['reservation_is_hard_upper_bound'] is False
