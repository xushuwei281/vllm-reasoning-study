"""Immutable artifacts for a neural proposal, before its outcome is observed."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .surrogate import NeuralSurrogate, connected_observation_groups, grouped_split, observation_key


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _target(row: Mapping, head: str) -> float | None:
    metrics = row["metrics"]
    if head == "p99_e2e_s" and (metrics.get("p99_censored") or metrics.get("p99_e2e_censored")):
        return None
    value = metrics.get(head)
    return float(value) if value is not None else None


def _identities(rows: Sequence[Mapping]) -> dict:
    return {"config_scenario_keys": sorted({observation_key(r) for r in rows}),
            **{plural: sorted({str(r[field]) for r in rows if r.get(field)})
               for field, plural in (("run_id", "run_ids"), ("trace_id", "trace_ids"), ("trace_hash", "trace_hashes"))}}


def _metrics(rows: Sequence[Mapping], predictions: Sequence[Mapping], baseline: Mapping) -> dict:
    result = {}
    for head, prediction_name in (("quality_goodput", "goodput"), ("p99_e2e_s", "p99")):
        pairs = [(value, p) for r, p in zip(rows, predictions) if (value := _target(r, head)) is not None]
        summary = {"count": len(pairs), "mae": None, "rmse": None,
                   "training_mean_baseline_mae": None, "training_mean_baseline_rmse": None,
                   "ensemble_two_std_coverage_descriptive": None}
        if pairs:
            actual = np.asarray([v for v, _ in pairs])
            error = np.asarray([p[prediction_name + "_mean"] - v for v, p in pairs])
            summary.update(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())),
                           ensemble_two_std_coverage_descriptive=float(np.mean([
                               abs(p[prediction_name + "_mean"] - v) <= 2 * p[prediction_name + "_std"]
                               for v, p in pairs])))
            if baseline[head] is not None:
                delta = actual - baseline[head]
                summary.update(training_mean_baseline_mae=float(np.abs(delta).mean()),
                               training_mean_baseline_rmse=float(np.sqrt((delta ** 2).mean())))
        result[head] = summary
    actual = np.asarray([float(bool(r["metrics"].get("feasible", False))) for r in rows])
    probability = np.asarray([p["feasible_probability"] for p in predictions])
    clipped = probability.clip(1e-7, 1 - 1e-7)
    bins = []
    for i in range(5):
        mask = (probability >= i / 5) & ((probability < (i + 1) / 5) if i < 4 else probability <= 1)
        bins.append({"lower": i / 5, "upper": (i + 1) / 5, "count": int(mask.sum()),
                     "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                     "observed_frequency": float(actual[mask].mean()) if mask.any() else None})
    result["feasibility"] = {
        "count": len(rows), "positive_count": int(actual.sum()), "negative_count": int(len(rows) - actual.sum()),
        "brier_score": float(((probability - actual) ** 2).mean()),
        "log_loss": float(-(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)).mean()),
        "training_prevalence_baseline_brier_score": float(((baseline["feasibility"] - actual) ** 2).mean()),
        "ece_five_bins_descriptive": float(sum(b["count"] * abs(b["mean_probability"] - b["observed_frequency"])
                                               for b in bins if b["count"]) / len(rows)),
        "reliability_bins": bins,
    }
    return result


def proposal_diagnostics(model: NeuralSurrogate, observed: Sequence[Mapping]) -> dict:
    """Describe the actual early-stop split; these are not external calibration tests."""
    train, validation = grouped_split(observed, model.settings["seed"])
    baseline = {}
    for head in ("quality_goodput", "p99_e2e_s"):
        values = [v for row in train if (v := _target(row, head)) is not None]
        baseline[head] = float(np.mean(values)) if values else None
    baseline["feasibility"] = float(np.mean([bool(r["metrics"].get("feasible", False)) for r in train]))
    splits = {}
    for name, rows in (("train", train), ("validation", validation)):
        predictions = model.predict(rows)
        groups = connected_observation_groups(rows)
        splits[name] = {"record_count": len(rows), "linked_group_count": len(groups),
                       "identities": _identities(rows), "linked_groups": [_identities(g) for g in groups],
                       "metrics": _metrics(rows, predictions, baseline),
                       "prediction_audit": [{"config_scenario_key": observation_key(r),
                                             **{k: r[k] for k in ("run_id", "trace_id", "trace_hash") if k in r},
                                             "actual_goodput": _target(r, "quality_goodput"),
                                             "actual_uncensored_p99": _target(r, "p99_e2e_s"),
                                             "actual_feasible": bool(r["metrics"].get("feasible", False)),
                                             "prediction": p} for r, p in zip(rows, predictions)]}
    caveats = ["Validation is the internal early-stopping split and was used for model selection; its errors are optimistic evidence, not an external held-out test.",
               "Ensemble standard deviation is model disagreement, not a calibrated confidence interval; coverage and ECE are descriptive only."]
    if len(connected_observation_groups(observed)) < 20:
        caveats.append("Fewer than 20 independent linked groups: a real fitted neural model is produced, but reliable generalization or superiority is not established.")
    if splits["validation"]["linked_group_count"] < 5:
        caveats.append("Fewer than five validation groups: ranking errors and calibration cannot be estimated reliably.")
    for name in splits:
        feasible = splits[name]["metrics"]["feasibility"]
        if not feasible["positive_count"] or not feasible["negative_count"]:
            caveats.append(f"The {name} split contains only one feasibility class; discrimination is not identified.")
        if splits[name]["metrics"]["p99_e2e_s"]["count"] < 2:
            caveats.append(f"The {name} split has fewer than two uncensored P99 labels; latency prediction is weakly supported.")
    if any(r["metrics"].get("completed", 0) < 1000 for r in observed):
        caveats.append("At least one run has fewer than 1,000 completed requests or lacks that count; run-level empirical P99 has weak tail support.")
    return {"training_only_baselines": baseline, "splits": splits,
            "parameters_per_ensemble_member": sum(p.numel() for p in model.models[0].parameters()),
            "caveats": caveats}


def persist_neural_proposal(directory: str | Path, *, ordinal: int, model: NeuralSurrogate,
                            observed: Sequence[Mapping], chosen: Mapping, decision: Mapping,
                            policy_settings: Mapping) -> dict:
    """Publish one self-contained, immutable, verifiable model directory atomically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parameter_hash = hashlib.sha256()
    for member in model.models:
        for name, tensor in member.state_dict().items():
            parameter_hash.update(name.encode())
            parameter_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    provenance = {"ordinal": ordinal, "source": model.report["source"],
                  "policy_settings": dict(policy_settings), "training_report": model.report,
                  "observed_sha256": _sha(_json(observed).encode()),
                  "parameters_sha256": parameter_hash.hexdigest(),
                  "selected_key": observation_key(chosen),
                  "selected_features": {k: chosen[k] for k in ("config_id", "scenario_id", "config", "workload")},
                  "selected_prediction": decision["prediction"], "acquisition": decision["acquisition"]}
    signature = _sha(_json(provenance).encode())
    destination = directory / f"proposal-{ordinal:03d}-{signature[:12]}"
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if (manifest["proposal_sha256"] != signature or
                set(manifest["files"]) != {"ensemble.pt", "observations.json", "report.json"} or any(
                _sha((destination / name).read_bytes()) != checksum for name, checksum in manifest["files"].items())):
            raise RuntimeError("Existing neural proposal artifact failed integrity verification")
    else:
        with tempfile.TemporaryDirectory(prefix=".neural-proposal-", dir=directory) as temporary:
            staging = Path(temporary)
            model.save(staging / "ensemble.pt")
            (staging / "observations.json").write_text(_json(observed))
            (staging / "report.json").write_text(_json({"provenance": provenance,
                                                       "diagnostics": proposal_diagnostics(model, observed)}))
            manifest = {"format_version": 1, "proposal_sha256": signature,
                        "files": {name: _sha((staging / name).read_bytes())
                                  for name in ("ensemble.pt", "observations.json", "report.json")}}
            (staging / "manifest.json").write_text(_json(manifest))
            staging.rename(destination)
    return {"directory": str(destination.resolve()), "checkpoint": str((destination / "ensemble.pt").resolve()),
            "report": str((destination / "report.json").resolve()), "proposal_sha256": signature}
