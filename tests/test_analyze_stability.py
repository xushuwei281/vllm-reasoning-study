"""Synthetic temporary fixtures validate read-only stability analysis."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_stability.py'
spec = importlib.util.spec_from_file_location('stability_test_module', SCRIPT)
stability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stability)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path, values):
    path.write_text(''.join(json.dumps(value) + '\n' for value in values))


def fixture_run(root, arm='default'):
    run_id = f'independent-{arm}-n150-seed724'
    path = root / 'results/continuation/runs' / run_id
    save(path / 'observation.json', {'source': 'measured', 'status': 'ok'})
    save(path / 'execution-protocol.json', {'warmup_s': 0, 'ramp_s': 0, 'measurement_s': 30})
    save(path / 'collection.json', {'window_start_s': 0, 'window_end_s': 30})
    requests = [{'request_id': 'r0', 'phase': 'measure', 'intended_arrival_s': 0, 'dispatch_s': .1,
                 'completion_s': 10, 'status': 'ok', 'finish_reason': 'stop'},
                {'request_id': 'r1', 'phase': 'measure', 'intended_arrival_s': 5, 'dispatch_s': 5.2,
                 'completion_s': 20, 'status': 'timeout', 'finish_reason': None},
                {'request_id': 'r2', 'phase': 'measure', 'intended_arrival_s': 15, 'dispatch_s': 15.1,
                 'completion_s': 35, 'status': 'ok', 'finish_reason': 'length'}]
    rows(path / 'requests.jsonl', requests)
    rows(path / 'offered.jsonl', [{k: row[k] for k in ('request_id', 'phase', 'intended_arrival_s')} for row in requests])
    rows(path / 'telemetry.jsonl', [
        {'time_s': stamp, 'device_memory_bytes': gib * 1024**3,
         'vllm:num_requests_waiting': waiting, 'vllm:num_requests_running': running}
        for stamp, gib, waiting, running in [(0, 1, 0, 1), (10, 2, 2, 2), (31, 3, 1, 1)]])
    return path


def test_reconstructs_queue_boundaries_and_keeps_memory_population_explicit(tmp_path):
    path = fixture_run(tmp_path)
    raw_before = {p.name: p.read_bytes() for p in path.iterdir()}
    result = stability.analyze(tmp_path, write_plots=False)
    default = result['arms'][0]
    assert result['status'] == 'partial' and result['arms'][1]['status'] == 'not_available'
    assert default['peak_client_outstanding'] == 2
    assert default['end_window_client_outstanding'] == 1
    assert default['last_recorded_client_outstanding'] == 0
    assert default['peak_device_memory_bytes'] == 2 * 1024**3
    assert default['peak_engine_waiting'] == default['peak_engine_running'] == 2
    assert default['error_count'] == default['truncated_count'] == 1
    assert default['client_dispatch_delay_p95_s'] == pytest.approx(.2)
    assert default['queue_slope_window_s'] == [0, 15]
    assert default['queue_slope_per_second'] is not None
    assert {p.name: p.read_bytes() for p in path.iterdir()} == raw_before


def test_missing_terminal_records_and_telemetry_are_unknown_not_false_zero(tmp_path):
    path = fixture_run(tmp_path)
    requests = stability.read_rows(path / 'requests.jsonl')
    rows(path / 'requests.jsonl', requests[:2])
    (path / 'telemetry.jsonl').unlink()
    result = stability.analyze(tmp_path, write_plots=False)['arms'][0]
    assert result['status'] == 'partial' and result['missing_terminal_records'] == 1
    assert result['peak_device_memory_bytes'] is None and result['peak_engine_waiting'] is None
    assert result['queue_slope_per_second'] is None
    assert any('undercount' in warning for warning in result['warnings'])


def test_no_synced_runs_produces_no_measured_plot_or_invented_values(tmp_path):
    result = stability.analyze(tmp_path)
    assert result['status'] == 'not_available' and result['plot_files'] == []
    assert all('peak_device_memory_bytes' not in row for row in result['arms'])


def test_truncated_jsonl_is_not_silently_repaired(tmp_path):
    path = fixture_run(tmp_path)
    raw = (path / 'requests.jsonl').read_bytes() + b'{"request_id":'
    (path / 'requests.jsonl').write_bytes(raw)
    with pytest.raises(json.JSONDecodeError):
        stability.analyze(tmp_path, write_plots=False)
    assert (path / 'requests.jsonl').read_bytes() == raw


def test_two_arm_plot_and_svg_are_generated_from_saved_trajectories(tmp_path):
    fixture_run(tmp_path, 'default')
    fixture_run(tmp_path, 'tuned')
    result = stability.analyze(tmp_path)
    assert result['status'] == 'complete'
    assert result['plot_files'] == ['independent_stability.png', 'independent_stability.svg']
    for name in result['plot_files']:
        assert (tmp_path / 'results/continuation/analysis' / name).stat().st_size > 1000
