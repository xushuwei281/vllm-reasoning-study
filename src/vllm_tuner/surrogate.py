"""Small CPU-trained neural performance surrogate; never backpropagates through vLLM.

Each record is one complete configuration × scenario × repetition. The feature
allowlist intentionally excludes realized output lengths and all evaluation outcomes.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

NUMERIC_FEATURES = (
    ("config", "max_num_batched_tokens"),
    ("config", "max_num_seqs"),
    ("config", "gpu_memory_utilization"),
    ("config", "tensor_parallel_size"),
    ("workload", "arrival_rate"),
    ("workload", "input_tokens_mean"),
    ("workload", "input_tokens_p95"),
    ("workload", "output_token_limit"),
    ("workload", "burstiness"),
)
CATEGORICAL_FEATURES = (("config", "precision"), ("workload", "profile"))


def observation_key(row: Mapping[str, Any]) -> str:
    return json.dumps([str(row["config_id"]), str(row["scenario_id"])], separators=(",", ":"))


def _numeric(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    try:
        x = np.asarray([[float(r[a][b]) for a, b in NUMERIC_FEATURES] for r in rows], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Numeric features must be present and resolved; capture omitted vLLM defaults from server startup") from error
    if not np.isfinite(x).all():
        raise ValueError("Predeployment numeric features must be finite")
    return x


class FeatureEncoder:
    """Training-only z scores and one-hot levels with a dedicated unseen bucket."""
    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "FeatureEncoder":
        if not rows:
            raise ValueError("Cannot fit preprocessing without training records")
        x = _numeric(rows)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        constant = self.scale < 1e-6
        # Unvarying count features (e.g. prompt length in a one-profile pilot) have
        # no learned extrapolation slope. A unit fallback would amplify a 100→350
        # token shift by 250 standard deviations. Use training magnitude only.
        self.scale[constant] = np.maximum(np.abs(self.mean[constant]), 1.0)
        self.categories = [sorted({str(r[a][b]) for r in rows}) for a, b in CATEGORICAL_FEATURES]
        return self

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not rows:
            return np.empty((0, self.dimension), dtype=np.float32)
        parts = [(_numeric(rows) - self.mean) / self.scale]
        for (a, b), levels in zip(CATEGORICAL_FEATURES, self.categories):
            index = {v: i for i, v in enumerate(levels)}
            block = np.zeros((len(rows), len(levels) + 1), dtype=np.float32)
            for i, row in enumerate(rows):
                block[i, index.get(str(row[a][b]), len(levels))] = 1.0
            parts.append(block)
        return np.concatenate(parts, axis=1).astype(np.float32)

    @property
    def dimension(self) -> int:
        return len(NUMERIC_FEATURES) + sum(len(c) + 1 for c in self.categories)

    def state_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "categories": self.categories}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureEncoder":
        obj = cls()
        obj.mean = np.asarray(state["mean"], dtype=np.float32)
        obj.scale = np.asarray(state["scale"], dtype=np.float32)
        obj.categories = state["categories"]
        return obj


def connected_observation_groups(rows: Sequence[Mapping[str, Any]]) -> list[list[dict]]:
    """Components linked by config/scenario, run ID or trace identity."""
    parents = list(range(len(rows)))
    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(rows):
        identities = [("config_scenario", observation_key(row))]
        identities.extend((field, str(row[field])) for field in ("run_id", "trace_id", "trace_hash") if row.get(field))
        for key in identities:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups: dict[int, list[dict]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(dict(row))
    return list(groups.values())


def grouped_split(rows: Sequence[Mapping[str, Any]], seed: int = 0,
                  validation_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Keep connected config/scenario, run and trace identities in one partition.

    Explicit development validation partitions are honored. A held-out workload
    must never be passed to fit; evaluation has a separate predict call.
    """
    if any(r.get("partition") in {"holdout", "test"} for r in rows):
        raise ValueError("Holdout/test rows are forbidden in surrogate fitting")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    source = {r.get("source", "measured") for r in rows}
    if len(source) > 1:
        raise ValueError("Synthetic and measured observations must not be mixed")
    groups = connected_observation_groups(rows)
    if len(groups) < 2:
        raise ValueError("Need at least two independent whole-run/config-scenario groups")
    if any(r.get("partition") == "validation" for r in rows):
        train, validation = [], []
        for group in groups:
            labels = {r.get("partition", "train") for r in group}
            if len(labels) != 1:
                raise ValueError("A linked run/trace/config-scenario crosses explicit partitions")
            (validation if labels == {"validation"} else train).extend(group)
        if not train or not validation:
            raise ValueError("Explicit split needs both training and validation groups")
        return train, validation
    keys = list(range(len(groups)))
    np.random.default_rng(seed).shuffle(keys)
    nval = min(len(keys) - 1, max(1, round(len(keys) * validation_fraction)))
    validation_keys = set(keys[:nval])
    return ([r for key in keys if key not in validation_keys for r in groups[key]],
            [r for key in keys if key in validation_keys for r in groups[key]])


def targets(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask unavailable/censored latency; zero goodput is a real log1p target."""
    y = np.zeros((len(rows), 2), dtype=np.float32)
    mask = np.zeros_like(y)
    feasible = np.zeros((len(rows), 1), dtype=np.float32)
    for i, row in enumerate(rows):
        metrics = row["metrics"]
        for j, name in enumerate(("quality_goodput", "p99_e2e_s")):
            value = metrics.get(name)
            if j == 1 and (metrics.get("p99_censored") or metrics.get("p99_e2e_censored")):
                value = None
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid nonnegative target {name}: {value}")
                y[i, j] = math.log1p(value)
                mask[i, j] = 1
        feasible[i, 0] = float(bool(metrics.get("feasible", False)))
    return torch.tensor(y), torch.tensor(mask), torch.tensor(feasible)


class LogTargetScaler:
    """Mask-aware training-only centering/scaling for the two log1p heads.

    Goodput values around .01 and latency values around hundreds of seconds
    otherwise produce very different regression residual magnitudes. Each
    observed training head is conditioned independently; no validation outcome
    enters the transformation. A constant/missing head receives a safe scale,
    but that does not make it statistically identified.
    """
    def fit(self, y: torch.Tensor, mask: torch.Tensor) -> "LogTargetScaler":
        if y.ndim != 2 or y.shape != mask.shape or y.shape[1] != 2:
            raise ValueError("two equally shaped target/mask heads required")
        self.center = torch.zeros(2, dtype=y.dtype)
        self.scale = torch.ones(2, dtype=y.dtype)
        self.counts = []
        for j in range(2):
            values = y[:, j][mask[:, j] > 0]
            self.counts.append(len(values))
            if len(values):
                self.center[j] = values.mean()
                spread = values.std(unbiased=False)
                self.scale[j] = spread if spread >= 1e-6 else max(abs(float(self.center[j])), 1.0)
        return self

    def transform(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Missing values remain neutral placeholders with zero loss weight.
        return torch.where(mask > 0, (y - self.center) / self.scale, torch.zeros_like(y))

    def inverse(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.scale + self.center

    def state_dict(self) -> dict[str, Any]:
        return {"center": self.center.tolist(), "scale": self.scale.tolist(), "counts": self.counts}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LogTargetScaler":
        obj = cls()
        obj.center = torch.tensor(state["center"], dtype=torch.float32)
        obj.scale = torch.tensor(state["scale"], dtype=torch.float32)
        obj.counts = list(state.get("counts", [0, 0]))
        if obj.center.shape != (2,) or obj.scale.shape != (2,) or not torch.isfinite(obj.center).all() or not torch.isfinite(obj.scale).all() or not (obj.scale > 0).all():
            raise ValueError("invalid target transformation checkpoint")
        return obj

    @classmethod
    def identity(cls) -> "LogTargetScaler":
        # Version-1 checkpoints predicted raw log1p targets directly.
        return cls.from_state_dict({"center": [0., 0.], "scale": [1., 1.], "counts": [0, 0]})


class PerformanceMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.05):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout))
        self.regression = nn.Linear(32, 2)
        self.feasibility = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.regression(h), self.feasibility(h)


def masked_loss(prediction: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor,
                mask: torch.Tensor, feasible: torch.Tensor) -> torch.Tensor:
    regression, logits = prediction
    terms = nn.functional.huber_loss(regression, y, reduction="none", delta=1.0)
    # Average each target over its observed entries so censoring does not implicitly
    # downweight latency as missingness rises. No imputed fast failure latencies.
    heads = (terms * mask).sum(0) / mask.sum(0).clamp(min=1)
    return heads.mean() + 0.25 * nn.functional.binary_cross_entropy_with_logits(logits, feasible)


class NeuralSurrogate:
    def __init__(self, seed: int = 0, ensemble_size: int = 3, max_epochs: int = 200,
                 patience: int = 20, learning_rate: float = 1e-3, dropout: float = 0.05,
                 standardize_targets: bool = True):
        if ensemble_size < 1 or max_epochs < 1 or patience < 1:
            raise ValueError("Positive ensemble_size, max_epochs and patience required")
        self.settings = dict(seed=seed, ensemble_size=ensemble_size, max_epochs=max_epochs,
                             patience=patience, learning_rate=learning_rate, dropout=dropout,
                             standardize_targets=standardize_targets)
        self.models: list[PerformanceMLP] = []
        self.report: dict[str, Any] = {}

    def fit(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        seed = self.settings["seed"]
        train, validation = grouped_split(observations, seed)
        self.encoder = FeatureEncoder().fit(train)
        tx = torch.tensor(self.encoder.transform(train))
        vx = torch.tensor(self.encoder.transform(validation))
        ty, tm, tf = targets(train)
        vy, vm, vf = targets(validation)
        self.target_scaler = (LogTargetScaler().fit(ty, tm) if self.settings["standardize_targets"]
                              else LogTargetScaler.identity())
        ty = self.target_scaler.transform(ty, tm)
        vy = self.target_scaler.transform(vy, vm)
        self.models, histories = [], []
        # CPU optimization avoids reserving an inference accelerator merely to train
        # a tiny table model. Run-level orchestration owns resource accounting.
        for member in range(self.settings["ensemble_size"]):
            torch.manual_seed(seed + member * 1009)
            model = PerformanceMLP(self.encoder.dimension, self.settings["dropout"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.settings["learning_rate"], weight_decay=1e-4)
            best, best_state, stale, best_epoch = float("inf"), None, 0, 0
            history = []
            for epoch in range(self.settings["max_epochs"]):
                model.train()
                optimizer.zero_grad()
                loss = masked_loss(model(tx), ty, tm, tf)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val = float(masked_loss(model(vx), vy, vm, vf))
                history.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "validation_loss": val})
                if val < best - 1e-6:
                    best, best_state, stale, best_epoch = val, copy.deepcopy(model.state_dict()), 0, epoch + 1
                else:
                    stale += 1
                if stale >= self.settings["patience"]:
                    break
            model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            histories.append({"best_epoch": best_epoch, "best_validation_loss": best, "epochs": history})
        independent_groups = len(connected_observation_groups(observations))
        self.report = {
            "source": observations[0].get("source", "measured"),
            "method_version": ("mlp-ensemble-v2-train-only-log-target-standardization"
                               if self.settings["standardize_targets"] else "mlp-ensemble-v1-unscaled-log-targets"),
            "target_transform": self.target_scaler.state_dict(),
            "training_records": len(train), "validation_records": len(validation),
            "training_groups": sorted({observation_key(r) for r in train}),
            "validation_groups": sorted({observation_key(r) for r in validation}),
            "independent_config_scenarios": len({observation_key(r) for r in observations}),
            "independent_linked_groups": independent_groups,
            "training_target_counts": {"goodput": int(tm[:, 0].sum()), "p99_e2e": int(tm[:, 1].sum())},
            "warning": "Too few independent points for reliable neural generalization; compare simpler models." if independent_groups < 20 else None,
            "feature_names": [".".join(f) for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            "members": histories,
        }
        return self.report

    def predict(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
        if not self.models:
            raise RuntimeError("Fit or load a surrogate before prediction")
        if not candidates:
            return []
        x = torch.tensor(self.encoder.transform(candidates))
        outputs, probabilities = [], []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                prediction, logits = model(x)
                log_prediction = self.target_scaler.inverse(prediction)
                outputs.append(torch.expm1(log_prediction.clamp(0, 20)).numpy())
                probabilities.append(torch.sigmoid(logits).numpy()[:, 0])
        array = np.stack(outputs)
        mean, std = array.mean(axis=0), array.std(axis=0, ddof=0)
        p = np.stack(probabilities).mean(axis=0)
        return [{"goodput_mean": float(mean[i, 0]), "goodput_std": float(std[i, 0]),
                 "p99_mean": float(mean[i, 1]), "p99_std": float(std[i, 1]),
                 "feasible_probability": float(p[i])} for i in range(len(candidates))]

    def save(self, path: str | Path) -> None:
        if not self.models:
            raise RuntimeError("Cannot checkpoint an unfitted surrogate")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format_version": 2, "settings": self.settings, "encoder": self.encoder.state_dict(),
                    "target_scaler": self.target_scaler.state_dict(), "models": [m.state_dict() for m in self.models], "report": self.report}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralSurrogate":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["format_version"] not in {1, 2}:
            raise ValueError("Unsupported checkpoint format")
        settings = dict(state["settings"])
        if state["format_version"] == 1:
            settings["standardize_targets"] = False
        obj = cls(**settings)
        obj.encoder = FeatureEncoder.from_state_dict(state["encoder"])
        obj.target_scaler = (LogTargetScaler.from_state_dict(state["target_scaler"])
                             if state["format_version"] == 2 else LogTargetScaler.identity())
        obj.report = state["report"]
        for parameters in state["models"]:
            model = PerformanceMLP(obj.encoder.dimension, obj.settings["dropout"])
            model.load_state_dict(parameters)
            model.eval()
            obj.models.append(model)
        return obj
