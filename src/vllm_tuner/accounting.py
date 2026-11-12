"""Append-only resource and result journals for one coordinated experiment queue."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping


class BudgetExceeded(RuntimeError):
    pass


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt journal at {path.name}:{lineno}; preserve it for audit") from exc
    return rows


def _append(path: Path, item: Mapping[str, Any]) -> None:
    serialized = json.dumps(item, sort_keys=True, allow_nan=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(serialized + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RunJournal:
    """Immutable one-row-per-run journal; repeated identical writes are idempotent."""
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        with _lock(self.path):
            return _read(self.path)

    def contains(self, run_id: str) -> bool:
        return any(row["run_id"] == run_id for row in self.records())

    def append(self, record: Mapping[str, Any]) -> bool:
        if not record.get("run_id"):
            raise ValueError("run_id required")
        normalized = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
        with _lock(self.path):
            for existing in _read(self.path):
                if existing["run_id"] == normalized["run_id"]:
                    if existing == normalized:
                        return False
                    raise ValueError("Conflicting record for immutable run_id")
            _append(self.path, normalized)
            return True


class BudgetLedger:
    """Admit bounded reservations, then charge actual allocated device-seconds.

    Startup, warmup, drain, shutdown, failed runs, and idle reserved devices are
    all part of wall_seconds. Optimizer CPU time is reported separately. If the
    accelerator stays reserved during training, its wall time stays inside the
    reservation. No request latency is used as a resource cost.

    A crash leaves an active reservation charged at its maximum until a human or
    recovery process supplies the actual wall time to finish(). This conservative
    hold avoids granting free budget after a restart.
    """
    def __init__(self, path: str | Path, total_accelerator_seconds: float = 28800,
                 validation_fraction: float = .25):
        if not math.isfinite(total_accelerator_seconds) or total_accelerator_seconds <= 0:
            raise ValueError("Positive finite budget required")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0,1)")
        self.path = Path(path)
        self.total = float(total_accelerator_seconds)
        self.reserve = self.total * validation_fraction
        policy = {"event": "policy", "total_accelerator_seconds": self.total,
                  "validation_fraction": validation_fraction}
        with _lock(self.path):
            existing = _read(self.path)
            if existing:
                if existing[0] != policy:
                    raise ValueError("Budget policy is immutable; use a new experiment ledger")
            else:
                _append(self.path, policy)

    @staticmethod
    def _state(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        starts, finishes = {}, {}
        for row in rows:
            if row["event"] == "begin":
                if row["run_id"] in starts:
                    raise ValueError("Duplicate budget begin event")
                starts[row["run_id"]] = row
            elif row["event"] == "finish":
                if row["run_id"] not in starts or row["run_id"] in finishes:
                    raise ValueError("Invalid budget finish event")
                finishes[row["run_id"]] = row
        return starts, finishes

    def _summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        starts, finishes = self._state(rows)
        used = sum(row["accelerator_seconds"] for row in finishes.values())
        active = [row for key, row in starts.items() if key not in finishes]
        held = sum(row["devices"] * row["max_wall_seconds"] for row in active)
        search_used = sum(finishes[key]["accelerator_seconds"] for key, row in starts.items()
                          if key in finishes and row["phase"] != "validation")
        search_held = sum(row["devices"] * row["max_wall_seconds"] for row in active if row["phase"] != "validation")
        return {"total_accelerator_seconds": self.total, "validation_reserved_seconds": self.reserve,
                "spent_accelerator_seconds": used, "held_accelerator_seconds": held,
                "remaining_accelerator_seconds": self.total - used - held,
                "remaining_search_accelerator_seconds": min(self.total - used - held,
                    self.total - self.reserve - search_used - search_held),
                "search_accelerator_seconds": search_used, "validation_accelerator_seconds": used - search_used,
                "optimizer_cpu_seconds": sum(row["optimizer_cpu_seconds"] for row in finishes.values()),
                "total_wall_seconds": sum(row["wall_seconds"] for row in finishes.values()),
                "active_run_ids": [row["run_id"] for row in active], "finished_runs": len(finishes),
                "overspent": used > self.total or search_used > self.total - self.reserve}

    def summary(self) -> dict[str, Any]:
        with _lock(self.path):
            return self._summary(_read(self.path))

    def begin(self, run_id: str, devices: int, max_wall_seconds: float,
              phase: str = "search") -> bool:
        if not run_id or not isinstance(devices, int) or isinstance(devices, bool) or devices < 0:
            raise ValueError("run_id and nonnegative integer allocated devices required")
        if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
            raise ValueError("Positive finite maximum reservation wall time required")
        if phase not in {"search", "calibration", "validation", "optimizer"}:
            raise ValueError("Unknown budget phase")
        with _lock(self.path):
            rows = _read(self.path)
            starts, finishes = self._state(rows)
            if run_id in starts:
                previous = starts[run_id]
                if (previous["devices"], previous["max_wall_seconds"], previous["phase"]) != (devices, max_wall_seconds, phase):
                    raise ValueError("Conflicting reservation for immutable run_id")
                return False
            # A single experiment queue prevents concurrent runs contaminating latency.
            if any(key not in finishes for key in starts):
                raise RuntimeError("Another reservation is active; recover/finish it before running")
            summary = self._summary(rows)
            available = summary["remaining_accelerator_seconds"] if phase == "validation" else summary["remaining_search_accelerator_seconds"]
            if devices * max_wall_seconds > available + 1e-9:
                raise BudgetExceeded("Reservation would spend the protected budget")
            _append(self.path, {"event": "begin", "run_id": run_id, "devices": devices,
                               "max_wall_seconds": max_wall_seconds, "phase": phase,
                               "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            return True

    def finish(self, run_id: str, wall_seconds: float, optimizer_cpu_seconds: float = 0,
               status: str = "ok") -> dict[str, Any]:
        if any(not math.isfinite(x) or x < 0 for x in (wall_seconds, optimizer_cpu_seconds)):
            raise ValueError("Costs must be finite and nonnegative")
        with _lock(self.path):
            rows = _read(self.path)
            starts, finishes = self._state(rows)
            if run_id not in starts:
                raise ValueError("Finish without an admitted reservation")
            event = {"event": "finish", "run_id": run_id, "wall_seconds": float(wall_seconds),
                     "optimizer_cpu_seconds": float(optimizer_cpu_seconds), "status": status,
                     "accelerator_seconds": starts[run_id]["devices"] * wall_seconds,
                     "reservation_overrun": wall_seconds > starts[run_id]["max_wall_seconds"]}
            if run_id in finishes:
                if finishes[run_id] != event:
                    raise ValueError("Conflicting cost for finished run_id")
                return finishes[run_id]
            _append(self.path, event)
            # Never hide an overrun by clipping its charge to the admitted bound.
            return event
