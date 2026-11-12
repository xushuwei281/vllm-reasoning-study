"""Sequential policies with a common observation interface and resumable choices.

This module never looks up an offline table. The caller must reveal only outcomes
already paid for by that method. A selected candidate is a proposal, not evidence
that its accuracy or latency constraints hold.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .surrogate import FeatureEncoder, NeuralSurrogate, observation_key


def measured_incumbent(observed: Sequence[Mapping[str, Any]]) -> dict | None:
    """Select by observed feasible mean goodput across available repetitions.

    Every repetition must be feasible. This conservative rule does not replace
    independent finalist validation or a paired accuracy noninferiority interval.
    """
    groups: dict[str, list[Mapping]] = {}
    for row in observed:
        groups.setdefault(observation_key(row), []).append(row)
    accepted = []
    for group in groups.values():
        if all(r["metrics"].get("feasible", False) and r["metrics"].get("quality_goodput") is not None for r in group):
            score = float(np.mean([r["metrics"]["quality_goodput"] for r in group]))
            if math.isfinite(score) and score >= 0:
                accepted.append((score, group[0]))
    if not accepted:
        return None
    score, row = max(accepted, key=lambda pair: pair[0])
    return {"config_id": row["config_id"], "scenario_id": row["scenario_id"], "quality_goodput": score}


def initial_design(candidates: Sequence[Mapping[str, Any]], n: int, seed: int = 0) -> list[dict]:
    """Space-filling common starts, using candidate descriptors only (no outcomes)."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    if not candidates or not n:
        return []
    if len({observation_key(c) for c in candidates}) != len(candidates):
        raise ValueError("Duplicate config-scenario candidates")
    x = FeatureEncoder().fit(candidates).transform(candidates)
    rng = np.random.default_rng(seed)
    default = next((i for i, c in enumerate(candidates) if c.get("is_default")), None)
    selected = [int(rng.integers(len(candidates))) if default is None else default]
    while len(selected) < min(n, len(candidates)):
        distance = ((x[:, None, :] - x[selected][None, :, :]) ** 2).sum(axis=2).min(axis=1)
        distance[selected] = -np.inf
        ties = np.flatnonzero(np.isclose(distance, distance.max()))
        selected.append(int(rng.choice(ties)))
    return [dict(candidates[i]) for i in selected]


class SearchPolicy:
    def __init__(self, method: str = "random", seed: int = 0, beta: float = 1.0,
                 p99_limit_s: float | None = None, neural_epochs: int = 200,
                 minimum_neural_groups: int = 6, default_config_id: str | None = None,
                 neural_target_standardization: bool = True):
        if method not in {"default", "random", "gp", "neural"}:
            raise ValueError("method must be default, random, gp or neural")
        if beta < 0 or (p99_limit_s is not None and p99_limit_s <= 0):
            raise ValueError("Invalid acquisition settings")
        self.settings = dict(method=method, seed=seed, beta=beta, p99_limit_s=p99_limit_s,
                             neural_epochs=neural_epochs, minimum_neural_groups=minimum_neural_groups,
                             default_config_id=default_config_id,
                             neural_target_standardization=neural_target_standardization)
        self.rng = np.random.default_rng(seed)
        self.selected: list[str] = []
        self.last_decision: dict[str, Any] = {}
        self.surrogate: NeuralSurrogate | None = None
        # Set by the orchestrator, outside settings/contract identity. Every
        # guided proposal then retains its own model, split and diagnostics.
        self.artifact_directory: Path | None = None

    def choose(self, candidates: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]) -> dict | None:
        started_cpu, started_wall = time.process_time(), time.perf_counter()
        if any(r.get("partition") in {"holdout", "test"} for r in observed):
            raise ValueError("Search cannot access held-out outcomes")
        if len({r.get("source", "measured") for r in observed}) > 1:
            raise ValueError("Search cannot mix synthetic and measured observations")
        if len({observation_key(c) for c in candidates}) != len(candidates):
            raise ValueError("Duplicate config-scenario candidates")
        used = set(self.selected) | {observation_key(r) for r in observed}
        remaining = [dict(c) for c in candidates if observation_key(c) not in used and c.get("known_valid", True)]
        if not remaining:
            return None
        method = self.settings["method"]
        decision: dict[str, Any] = {"requested_method": method, "observations_seen": len(observed),
                                    "candidate_count": len(remaining)}
        if method == "default":
            chosen = next((c for c in remaining if c.get("is_default") or c["config_id"] == self.settings["default_config_id"]), None)
            if chosen is None:
                return None  # A fixed default is not allowed to become another search.
            decision["method_used"] = "default"
        elif method == "random" or len({observation_key(r) for r in observed}) < 2:
            chosen = remaining[int(self.rng.integers(len(remaining)))]
            decision["method_used"] = "random" if method == "random" else "random_insufficient_observations"
        elif method == "neural" and len({observation_key(r) for r in observed}) < self.settings["minimum_neural_groups"]:
            chosen = remaining[int(self.rng.integers(len(remaining)))]
            decision["method_used"] = "random_neural_warm_start"
        else:
            if method == "neural":
                scores, details = self._neural_scores(remaining, observed)
            else:
                scores, details = self._gp_scores(remaining, observed)
            # Stable seeded tie-breaking, including all-zero expected improvement.
            best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-10, atol=1e-12))
            index = int(self.rng.choice(best))
            chosen = remaining[index]
            decision.update(method_used=method, acquisition=float(scores[index]), prediction=details[index])
        key = observation_key(chosen)
        if decision["method_used"] == "neural" and self.artifact_directory is not None:
            from .neural_artifacts import persist_neural_proposal
            decision["model_artifact"] = persist_neural_proposal(
                self.artifact_directory, ordinal=len(self.selected) + 1, model=self.surrogate,
                observed=observed, chosen=chosen, decision=decision, policy_settings=self.settings)
        self.selected.append(key)
        decision.update(selected_key=key, optimizer_cpu_seconds=time.process_time() - started_cpu,
                        optimizer_wall_seconds=time.perf_counter() - started_wall)
        self.last_decision = decision
        return chosen

    def _neural_scores(self, remaining: list[dict], observed: Sequence[Mapping]) -> tuple[np.ndarray, list[dict]]:
        self.surrogate = NeuralSurrogate(seed=self.settings["seed"], max_epochs=self.settings["neural_epochs"],
                                         standardize_targets=self.settings["neural_target_standardization"])
        self.surrogate.fit(observed)
        predictions = self.surrogate.predict(remaining)
        scores = []
        for p in predictions:
            score = p["feasible_probability"] * (p["goodput_mean"] + self.settings["beta"] * p["goodput_std"])
            if self.settings["p99_limit_s"] is not None:
                # Ranking heuristic only: ensemble spread is not a calibrated CI.
                spread = max(p["p99_std"], 0.05 * max(p["p99_mean"], 1e-3))
                score *= float(norm.cdf((self.settings["p99_limit_s"] - p["p99_mean"]) / spread))
            scores.append(score)
        return np.asarray(scores), predictions

    def _gp_scores(self, remaining: list[dict], observed: Sequence[Mapping]) -> tuple[np.ndarray, list[dict]]:
        encoder = FeatureEncoder().fit(observed)
        x, query = encoder.transform(observed), encoder.transform(remaining)
        # Conventional constrained GP expected improvement in log1p-goodput space.
        # Infeasible trials receive utility zero; failures remain visible.
        labels = np.asarray([bool(r["metrics"].get("feasible", False)) for r in observed])
        y = np.asarray([math.log1p(max(0.0, float(r["metrics"].get("quality_goodput") or 0.0)))
                        if ok else 0.0 for r, ok in zip(observed, labels)])
        kernel = ConstantKernel(1.0, (0.01, 100)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.01, (1e-5, 1))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                      random_state=self.settings["seed"], n_restarts_optimizer=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x, y)
            mean, std = gp.predict(query, return_std=True)
            if len(np.unique(labels)) == 2:
                feasibility = GaussianProcessClassifier(1.0 * Matern(nu=2.5), random_state=self.settings["seed"], max_iter_predict=50)
                feasibility.fit(x, labels)
                p_feasible = feasibility.predict_proba(query)[:, list(feasibility.classes_).index(True)]
            else:
                p_feasible = np.full(len(query), (float(labels.sum()) + 0.5) / (len(labels) + 1))
        best = float(y[labels].max()) if labels.any() else 0.0
        std = np.maximum(std, 1e-9)
        improvement = mean - best
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        score = np.maximum(0.0, ei) * p_feasible
        predictions = [{"log_goodput_mean": float(m), "log_goodput_std": float(s),
                        "feasible_probability": float(p), "expected_improvement": float(e)}
                       for m, s, p, e in zip(mean, std, p_feasible, ei)]
        if self.settings["p99_limit_s"] is not None:
            valid = [i for i, r in enumerate(observed) if r["metrics"].get("p99_e2e_s") is not None
                     and not r["metrics"].get("p99_censored", False)
                     and not r["metrics"].get("p99_e2e_censored", False)]
            if len(valid) >= 2:
                latency = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                                    random_state=self.settings["seed"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    latency.fit(x[valid], np.log1p([observed[i]["metrics"]["p99_e2e_s"] for i in valid]))
                    lm, ls = latency.predict(query, return_std=True)
                constraint = norm.cdf((math.log1p(self.settings["p99_limit_s"]) - lm) / np.maximum(ls, 1e-9))
                score *= constraint
                for p, probability in zip(predictions, constraint):
                    p["latency_constraint_probability"] = float(probability)
        return score, predictions

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"format_version": 1, "settings": self.settings, "rng_state": self.rng.bit_generator.state,
                 "selected": self.selected, "last_decision": self.last_decision}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        if self.surrogate is not None:
            self.surrogate.save(path.with_suffix(path.suffix + ".pt"))

    @classmethod
    def load(cls, path: str | Path) -> "SearchPolicy":
        state = json.loads(Path(path).read_text())
        if state["format_version"] != 1:
            raise ValueError("Unsupported search checkpoint")
        settings = dict(state["settings"])
        # Old policy files belong to the original unscaled-target experiment.
        settings.setdefault("neural_target_standardization", False)
        obj = cls(**settings)
        obj.rng.bit_generator.state = state["rng_state"]
        obj.selected, obj.last_decision = state["selected"], state["last_decision"]
        # Refits are deterministic from the caller's immutable observed ledger.
        return obj
