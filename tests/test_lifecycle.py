import sys
import time
import pytest
from vllm_tuner.lifecycle import ServerProcess


def test_lifecycle_counts_reserved_idle_and_shutdown(tmp_path):
    process = ServerProcess([sys.executable, '-c', 'import time; time.sleep(20)'],
                            tmp_path/'server.log', allocated_devices=2)
    with process:
        time.sleep(.02)
        assert process.cost()['accelerator_seconds'] > 0
    first = process.cost()
    process.stop()
    assert process.process.poll() is not None
    assert first == process.cost()
    assert first['accelerator_seconds'] == 2 * first['wall_seconds']


def test_spawn_failure_is_charged_and_log_preserved(tmp_path):
    process = ServerProcess(['/not/a/real/executable'], tmp_path/'server.log', allocated_devices=1)
    with pytest.raises(FileNotFoundError):
        process.start()
    assert process.cost()['accelerator_seconds'] >= 0
    assert process.cost()['lifecycle_status'] == 'startup_failed'
    assert (tmp_path/'server.log').exists()


def test_readiness_crash_is_charged(tmp_path):
    process = ServerProcess([sys.executable, '-c', 'raise SystemExit(7)'], tmp_path/'failed.log',
                            allocated_devices=1, startup_timeout_s=.5,
                            health_url='http://127.0.0.1:59317/health')
    with pytest.raises(RuntimeError):
        process.start()
    assert process.cost()['accelerator_seconds'] > 0
    assert process.cost()['lifecycle_status'] == 'startup_failed'


def test_log_cannot_be_overwritten(tmp_path):
    path = tmp_path/'server.log'; path.write_text('evidence')
    process = ServerProcess([sys.executable, '-c', 'pass'], path, allocated_devices=0)
    with pytest.raises(FileExistsError):
        process.start()
    assert path.read_text() == 'evidence'


def test_readiness_timeout_stops_process_and_counts_failed_reservation(tmp_path):
    process = ServerProcess([sys.executable, '-c', 'import time; time.sleep(20)'],
                            tmp_path/'timeout.log', allocated_devices=1,
                            startup_timeout_s=.12, health_url='http://127.0.0.1:59318/health')
    with pytest.raises(TimeoutError):
        process.start()
    assert process.process.poll() is not None
    assert process.cost()['wall_seconds'] >= .12
    assert process.cost()['lifecycle_status'] == 'startup_failed'
