"""Owned subprocess lifecycle and reserved-device accounting, including failures."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Mapping, Sequence


class ServerProcess:
    """Start a server in its own process group and always reap owned children.

    Reservation begins immediately before spawn and ends after group shutdown,
    so compilation, warmup, crashes, idle time and restarts are charged. The
    caller must additionally charge any externally reserved time between runs.
    """
    def __init__(self, command: Sequence[str], log_path: str | Path,
                 allocated_devices: int, startup_timeout_s: float = 180,
                 shutdown_timeout_s: float = 15, health_url: str | None = None,
                 env: Mapping[str, str] | None = None):
        if allocated_devices < 0 or startup_timeout_s <= 0 or shutdown_timeout_s <= 0:
            raise ValueError('invalid reservation count or timeout')
        self.command = list(command)
        self.log_path = Path(log_path)
        self.allocated_devices = allocated_devices
        self.startup_timeout_s = startup_timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s
        self.health_url = health_url
        self.env = dict(env) if env is not None else None
        self.process: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.ready_at: float | None = None
        self.status = 'not_started'
        self._log = None

    def start(self) -> 'ServerProcess':
        if self.started_at is not None:
            raise RuntimeError('ServerProcess objects are single-use; record restarts as new reservations')
        if self.health_url:
            from urllib.parse import urlsplit
            parsed = urlsplit(self.health_url)
            # A healthy old process must not be mistaken for the new server.
            if parsed.hostname in ('127.0.0.1', 'localhost'):
                with socket.socket() as sock:
                    if sock.connect_ex((parsed.hostname, parsed.port or 80)) == 0:
                        raise RuntimeError('requested port is already in use; refusing ambiguous readiness')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open('x')
        self.started_at = time.monotonic()
        self.status = 'starting'
        try:
            self.process = subprocess.Popen(self.command, stdout=self._log, stderr=subprocess.STDOUT,
                                            start_new_session=True, env=self.env)
            if self.health_url:
                self.wait_ready()
            else:
                self.status = 'running'
            return self
        except BaseException:
            self.status = 'startup_failed'
            self.stop()
            raise

    def wait_ready(self) -> None:
        assert self.started_at is not None and self.process is not None
        while time.monotonic() - self.started_at < self.startup_timeout_s:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f'server exited before ready with code {code}; see {self.log_path}')
            try:
                with urllib.request.urlopen(self.health_url, timeout=min(2, self.startup_timeout_s)) as response:
                    if response.status == 200:
                        self.ready_at = time.monotonic()
                        self.status = 'running'
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        raise TimeoutError(f'server readiness exceeded {self.startup_timeout_s}s')

    def stop(self) -> None:
        if self.stopped_at is not None:
            return
        if self.process:
            # Kill the owned process group even when its leader has already
            # exited: worker descendants can otherwise retain device memory.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=self.shutdown_timeout_s)
            # Ensure orphaned workers from an already-dead leader are stopped.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.started_at is not None:
            self.stopped_at = time.monotonic()
        if self._log:
            self._log.close()
        if self.status not in ('startup_failed', 'not_started'):
            self.status = 'stopped'

    def cost(self) -> dict:
        elapsed = 0.0 if self.started_at is None else (self.stopped_at or time.monotonic()) - self.started_at
        return {'wall_seconds': elapsed, 'accelerator_seconds': elapsed * self.allocated_devices,
                'allocated_devices': self.allocated_devices, 'optimizer_cpu_seconds': 0.0,
                'startup_seconds': None if self.ready_at is None else self.ready_at - self.started_at,
                'lifecycle_status': self.status}

    def __enter__(self) -> 'ServerProcess':
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
