"""CPU-only stability diagnostics from saved independent-validation raw records.

No inference, test answers, remote calls or immutable-record writes. Missing
telemetry remains missing; one trajectory does not establish reproducibility.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from vllm_tuner.evaluation import queue_trace, percentile


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_rows(path):
    if not path.exists():
        return []
    # Fail visibly on a partial JSONL line; do not silently repair raw evidence.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def last_state(rows, cutoff):
    available = [row for row in rows if finite(row.get('time_s')) and row['time_s'] <= cutoff]
    return available[-1]['client_outstanding'] if available else 0


def grid_slope(rows, start, end):
    """Descriptive OLS drift on a uniform one-second grid, not an inferential CI."""
    if end - start < 2 or not rows:
        return None
    times = [row['time_s'] for row in rows]
    grid = [start + i for i in range(math.floor(end - start) + 1)]
    values = [rows[index]['client_outstanding'] if (index := bisect_right(times, stamp) - 1) >= 0 else 0
              for stamp in grid]
    mean_x, mean_y = sum(grid) / len(grid), sum(values) / len(values)
    denominator = sum((stamp - mean_x) ** 2 for stamp in grid)
    return sum((stamp - mean_x) * (value - mean_y) for stamp, value in zip(grid, values)) / denominator


def arm_diagnostics(root, arm, run_id):
    directory = root / 'results/continuation/runs' / run_id
    names = ['observation.json', 'execution-protocol.json', 'plan.json', 'collection.json',
             'requests.jsonl', 'offered.jsonl', 'queue.jsonl', 'telemetry.jsonl']
    files = [directory / name for name in names if (directory / name).exists()]
    base = {'arm': arm, 'run_id': run_id, 'raw_paths': [str(p.relative_to(root)) for p in files],
            'input_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    if not files:
        return {**base, 'status': 'not_available', 'warnings': ['Independent run artifacts are not synchronized']}, None
    observation = read_json(directory / 'observation.json')
    if observation and observation.get('source') != 'measured':
        raise ValueError('Stability plots accept measured independent records only')
    collection = read_json(directory / 'collection.json')
    collector_plan = read_json(directory / 'plan.json') or {}
    protocol = read_json(directory / 'execution-protocol.json') or collector_plan.get('protocol')
    if not protocol:
        return {**base, 'status': 'partial', 'warnings': ['No recorded execution protocol; no window inferred']}, None
    start = protocol['warmup_s'] + protocol['ramp_s']
    end = start + protocol['measurement_s']
    if collection and (collection['window_start_s'] != start or collection['window_end_s'] != end):
        raise ValueError('Collection and recorded execution window disagree')
    requests = read_rows(directory / 'requests.jsonl')
    offered = read_rows(directory / 'offered.jsonl')
    telemetry = read_rows(directory / 'telemetry.jsonl')
    queue = read_rows(directory / 'queue.jsonl') if (directory / 'queue.jsonl').exists() else queue_trace(requests)
    queue = sorted(queue, key=lambda row: row['time_s'])
    cohort = [r for r in requests if r.get('phase') == 'measure'
              and finite(r.get('intended_arrival_s')) and start <= r['intended_arrival_s'] < end]
    measured_offered = [r for r in offered if r.get('phase') == 'measure']
    arrivals = [r['intended_arrival_s'] for r in measured_offered if finite(r.get('intended_arrival_s'))]
    last_arrival = max(arrivals) if arrivals else None
    drift_end = min(end, last_arrival) if last_arrival is not None else end
    observed_ids = {r['request_id'] for r in requests}
    missing_terminal = sum(r['request_id'] not in observed_ids for r in offered)
    memory = [r for r in telemetry if finite(r.get('time_s')) and finite(r.get('device_memory_bytes'))]
    def peak(field):
        values = [r[field] for r in telemetry if finite(r.get('time_s')) and start <= r['time_s'] <= end
                  and finite(r.get(field))]
        return max(values) if values else None
    memory_window = [r['device_memory_bytes'] for r in memory if start <= r['time_s'] <= end]
    queue_window = [r['client_outstanding'] for r in queue if start <= r['time_s'] <= end]
    delays = [r['dispatch_s'] - r['intended_arrival_s'] for r in cohort if finite(r.get('dispatch_s'))]
    warnings = []
    if not observation or not collection:
        warnings.append('Incomplete collector/observation artifacts; snapshot is partial')
    if observation and observation.get('status') != 'ok':
        warnings.append('Run status is ' + str(observation.get('status')))
    if not telemetry:
        warnings.append('Telemetry not saved; device memory/engine trajectories cannot be reconstructed')
    if not memory:
        warnings.append('Whole-device GPU memory observations are unavailable; no RSS substitution')
    if missing_terminal:
        warnings.append(f'{missing_terminal} offered requests lack terminal records; reconstructed queue can undercount outstanding work')
    result = {**base, 'status': 'complete' if not warnings else 'partial',
              'measurement_window_s': [start, end], 'last_offered_arrival_s': last_arrival,
              'recorded_requests': len(requests), 'measurement_requests': len(cohort),
              'offered_measurement_requests': len(measured_offered), 'missing_terminal_records': missing_terminal,
              'telemetry_samples': len(telemetry),
              'peak_device_memory_bytes': max(memory_window) if memory_window else None,
              'peak_client_outstanding': max(queue_window) if queue_window else None,
              'peak_engine_waiting': peak('vllm:num_requests_waiting'),
              'peak_engine_running': peak('vllm:num_requests_running'),
              'end_window_client_outstanding': last_state(queue, end) if queue else None,
              'last_recorded_client_outstanding': queue[-1]['client_outstanding'] if queue else None,
              'error_count': sum(r.get('status') != 'ok' for r in cohort),
              'truncated_count': sum(r.get('finish_reason') == 'length' for r in cohort),
              'client_dispatch_delay_p95_s': percentile(delays, .95) if delays else None,
              'queue_slope_per_second': grid_slope(queue, start, drift_end) if not missing_terminal else None,
              'queue_slope_window_s': [start, drift_end],
              'queue_slope_note': 'Descriptive OLS slope on a one-second grid through last offered arrival; excludes the quiet tail and is not a stability test or confidence interval',
              'warnings': warnings}
    return result, {'arm': arm, 'queue': queue, 'telemetry': telemetry, 'memory': memory,
                    'window': [start, end], 'last_arrival': last_arrival}


def create_plot(series, destination):
    if not series:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    colors = {'default': '#377eb8', 'tuned': '#e58320'}
    fig, (queue_ax, memory_ax) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    latest = max([s['window'][1] for s in series]
                 + [r['time_s'] for s in series for r in s['queue']]
                 + [r['time_s'] for s in series for r in s['telemetry'] if finite(r.get('time_s'))])
    for item in series:
        label, color = item['arm'], colors[item['arm']]
        if item['queue']:
            queue_ax.step([r['time_s'] for r in item['queue']], [r['client_outstanding'] for r in item['queue']],
                          where='post', color=color, linewidth=1.6, label=label + ': client outstanding')
        waiting = [r for r in item['telemetry'] if finite(r.get('time_s')) and finite(r.get('vllm:num_requests_waiting'))]
        if waiting:
            queue_ax.step([r['time_s'] for r in waiting], [r['vllm:num_requests_waiting'] for r in waiting],
                          where='post', color=color, linestyle='--', alpha=.8, label=label + ': engine waiting')
        if item['memory']:
            memory_ax.plot([r['time_s'] for r in item['memory']],
                           [r['device_memory_bytes'] / 1024**3 for r in item['memory']],
                           color=color, linewidth=1.6, label=label)
    window_end = max(s['window'][1] for s in series)
    last_arrivals = [s['last_arrival'] for s in series if s['last_arrival'] is not None]
    for axis in (queue_ax, memory_ax):
        if last_arrivals:
            axis.axvline(max(last_arrivals), color='#777777', linestyle=':', alpha=.7)
        axis.axvline(window_end, color='#555555', linestyle='--', alpha=.5)
        if latest > window_end:
            axis.axvspan(window_end, latest, color='#e8d4b5', alpha=.25)
        axis.grid(alpha=.2)
        axis.set_xlim(0, max(latest, window_end))
        if axis.lines:
            axis.legend(loc='best', fontsize=8)
    if not any(s['memory'] for s in series):
        memory_ax.text(.5, .5, 'Device-memory telemetry unavailable', ha='center', transform=memory_ax.transAxes)
    queue_ax.set(ylabel='Requests', title='Independent validation: queue and allocated-device memory observations')
    memory_ax.set(ylabel='Sampled whole-device memory (GiB)', xlabel='Seconds after collection begins (startup excluded)')
    fig.text(.5, .018, '150-question finite cohort at 0.6/s spacing; dotted line: last arrival; dashed line: measurement end.\n'
             'Shaded region: drain. One run per arm; descriptive trajectories do not establish deployment reproducibility.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, 1))
    files = []
    for suffix in ('png', 'svg'):
        path = destination / ('independent_stability.' + suffix)
        fig.savefig(path, dpi=180)
        files.append(path.name)
    plt.close(fig)
    return files


def analyze(root, *, write_plots=True):
    root = Path(root)
    output = root / 'results/continuation'
    evidence = read_json(output / 'independent-validation-evidence.json') or {}
    arms, series = [], []
    for arm in ('default', 'tuned'):
        run_ids = evidence.get(arm, {}).get('run_ids', [f'independent-{arm}-n150-seed724'])
        if len(run_ids) != 1:
            raise ValueError('This preregistered diagnostic expects one independent run per arm')
        if Path(run_ids[0]).name != run_ids[0]:
            raise ValueError('Invalid independent run identifier')
        summary, trajectory = arm_diagnostics(root, arm, run_ids[0])
        arms.append(summary)
        if trajectory:
            series.append(trajectory)
    destination = output / 'analysis'
    destination.mkdir(parents=True, exist_ok=True)
    status = ('complete' if all(a['status'] == 'complete' for a in arms)
              else 'not_available' if all(a['status'] == 'not_available' for a in arms) else 'partial')
    result = {'schema_version': 1, 'status': status, 'source': 'measured',
              'phase': 'independent_development_validation', 'arms': arms,
              'limitations': ['One run per arm; no between-run or robust P99 claim.',
                              'Client outstanding includes service and network time; engine waiting is a distinct sampled metric.',
                              'GPU memory is sampled whole-device usage, not model-only memory or active compute utilization.',
                              'Independent validation is a finite cohort with a quiet tail, not 300 seconds of constant .6/s arrivals.'],
              'plot_files': create_plot(series, destination) if write_plots else [],
              'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (destination / 'stability-summary.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    result = analyze(args.root, write_plots=not args.no_plots)
    print(json.dumps({'status': result['status'], 'plot_files': result['plot_files']}))


if __name__ == '__main__':
    main()
