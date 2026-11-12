"""CPU-only integrity checks: fake lifecycle and stream, no model/server launch."""
import copy
import json
import pytest
from vllm_tuner.accounting import BudgetExceeded, RunJournal
from vllm_tuner.config import BASE_CONFIG, MODEL
from vllm_tuner import experiment


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    class FakeServer:
        starts = 0
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            FakeServer.starts += 1
        def stop(self):
            pass
        def cost(self):
            return {'wall_seconds': 2.0, 'accelerator_seconds': 2.0,
                    'optimizer_cpu_seconds': 0.0, 'allocated_devices': 1}

    async def fake_collect(*args, **kwargs):
        rows = [{'request_id': 'r0', 'question_id': 'q0', 'phase': 'measure',
                 'intended_arrival_s': .1, 'dispatch_s': .1, 'first_token_s': .2,
                 'completion_s': .3, 'status': 'ok', 'text': '<think>one</think>\\boxed{1}',
                 'input_tokens': 3, 'output_tokens': 4, 'finish_reason': 'stop', 'chunks': []}]
        return rows, [], {'window_start_s': 0., 'window_end_s': 1., 'trace_hash': 'unit-trace'}

    monkeypatch.setattr(experiment, 'ServerProcess', FakeServer)
    monkeypatch.setattr(experiment, 'collect', fake_collect)
    study = {'model': copy.deepcopy(MODEL), 'startup_timeout_s': 1,
             'budget': {'total_accelerator_seconds': 1000, 'validation_fraction': .25, 'allocated_devices': 1},
             'benchmark': {'warmup_s': 0., 'ramp_s': 0., 'measurement_s': 1., 'drain_s': 1.,
                           'timeout_s': 1., 'max_in_flight': 4},
             'decoding': {'temperature': .6, 'seed': 0},
             'slos': {'ttft_s': 1., 'tpot_s': 1., 'e2e_s': 1.},
             'constraints': {'max_accuracy_drop': .02, 'max_error_rate': .01}}
    caps = {'device_count': 1, 'allowed_tp_sizes': [1], 'supported_precisions': ['bf16'],
            'chunked_prefill': True, 'prefix_caching': True, 'max_model_len': 4096,
            'max_memory_fraction': .65}
    candidate = {'config_id': 'unit-config', 'scenario_id': 'unit-scenario',
                 'config': copy.deepcopy(BASE_CONFIG), 'workload': {'arrival_rate': 1., 'output_token_limit': 1024,
                 'input_tokens_max': 3, 'profile': 'short'}}
    values = dict(candidate=candidate, study=study, capabilities=caps, environment={'environment_hash': 'unit'},
                  questions=[{'question_id': 'q0', 'question': 'one?', 'gold': '#### 1'}], prompts=['unit prompt'],
                  output_root=tmp_path/'out', executable='/fake/vllm', run_id='unit-1', reference_accuracy=1.,
                  runtime_env={'UNIT_TEST': '1'}, ledger_path=tmp_path/'global-budget.jsonl')
    return values, FakeServer


def test_identical_reuse_is_idempotent_changed_config_rejected(inputs):
    values, fake = inputs
    original = experiment.run_trial(**values)
    assert original['status'] == 'ok'
    assert experiment.run_trial(**values) == original
    assert fake.starts == 1
    changed = copy.deepcopy(values)
    changed['candidate']['config']['max_num_seqs'] += 1
    with pytest.raises(ValueError, match='stale reuse'):
        experiment.run_trial(**changed)
    assert fake.starts == 1


@pytest.mark.parametrize('change', ['gold', 'deadline', 'environment', 'repetition', 'capability'])
def test_all_result_affecting_inputs_invalidate_run_id(inputs, change):
    values, fake = inputs
    values['trace'] = [{'phase': 'measure', 'intended_arrival_s': .1}]
    experiment.run_trial(**values)
    changed = copy.deepcopy(values)
    if change == 'gold': changed['questions'][0]['gold'] = '#### 2'
    if change == 'deadline': changed['deadline_mode'] = True
    if change == 'environment': changed['runtime_env']['VLLM_METAL_BACKEND'] = 'mps'
    if change == 'repetition': changed['repetition'] = 1
    if change == 'capability': changed['capabilities']['max_memory_fraction'] = .6
    with pytest.raises(ValueError, match='stale reuse'):
        experiment.run_trial(**changed)
    assert fake.starts == 1


@pytest.mark.parametrize('trace', [
    [{'phase': 'measure', 'intended_arrival_s': 2.}],
    [{'phase': 'measure', 'intended_arrival_s': float('inf')}],
    [{'phase': 'measure', 'intended_arrival_s': .8}, {'phase': 'measure', 'intended_arrival_s': .2}],
])
def test_unbounded_or_unsorted_trace_does_not_reserve_or_create_run(inputs, trace):
    values, fake = inputs
    with pytest.raises(ValueError):
        experiment.run_trial(**values, trace=trace)
    assert not values['output_root'].exists()
    assert not values['ledger_path'].exists()
    assert fake.starts == 0


def test_denied_budget_has_no_run_directory(inputs):
    values, fake = inputs
    values['study']['budget']['total_accelerator_seconds'] = 1
    with pytest.raises(BudgetExceeded):
        experiment.run_trial(**values)
    assert not (values['output_root']/'runs'/'unit-1').exists()
    assert fake.starts == 0
    events = [json.loads(line) for line in values['ledger_path'].read_text().splitlines()]
    assert [event['event'] for event in events] == ['policy']


def test_known_invalid_is_recorded_without_device_time(inputs):
    values, fake = inputs
    values['candidate']['config']['tensor_parallel_size'] = 2
    result = experiment.run_trial(**values)
    assert result['status'] == 'known_invalid'
    assert result['metrics']['feasible'] is False
    assert result['cost']['accelerator_seconds'] == 0
    assert fake.starts == 0
    assert RunJournal(values['output_root']/'observations.jsonl').records() == [result]
    assert (values['output_root']/'runs'/'unit-1'/'observation.json').exists()


def test_measured_accelerator_run_cannot_claim_zero_allocated_devices(inputs):
    values, fake = inputs
    values['study']['budget']['allocated_devices'] = 0
    with pytest.raises(ValueError, match='positive allocated device'):
        experiment.run_trial(**values)
    assert fake.starts == 0


def test_bad_cohort_cannot_reserve_hardware(inputs):
    values, fake = inputs
    with pytest.raises(ValueError, match='explicit arrival'):
        experiment.run_trial(**values, deadline_mode=True, trace=[])
    assert not values['ledger_path'].exists()
    assert fake.starts == 0


@pytest.mark.parametrize('rate', [float('inf'), float('nan'), 0, -1])
def test_generated_arrivals_cannot_hang_on_invalid_rate(inputs, rate):
    values, fake = inputs
    values['candidate']['workload']['arrival_rate'] = rate
    with pytest.raises(ValueError, match='arrival_rate'):
        experiment.run_trial(**values)
    assert not values['ledger_path'].exists()
    assert fake.starts == 0


def test_fractional_concurrency_and_busy_loop_monitor_rejected(inputs):
    values, fake = inputs
    with pytest.raises(ValueError, match='max_in_flight'):
        experiment.run_trial(**values, protocol_override={'max_in_flight': 1.5})
    with pytest.raises(ValueError, match='telemetry_interval'):
        experiment.run_trial(**values, protocol_override={'telemetry_interval_s': 0})
    assert fake.starts == 0
