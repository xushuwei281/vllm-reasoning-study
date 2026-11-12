"""Independent development validation, then explicitly staged held-out testing.

Run ``python scripts/hf_validation.py validate`` after all three live searches.
Run ``python scripts/hf_validation.py final`` only after validation creates the
immutable freeze. Importing this module never opens test data or starts serving.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import hf_continuation as continuation
from vllm_tuner.data import (digest, freeze_protocol, prepare_dataset, prompt_for,
                             read_jsonl, select_questions, write_json)
from vllm_tuner.statistics import (EXACT_NONINFERIORITY_METHOD, paired_accuracy_ci,
                                   paired_accuracy_noninferiority)

METHODS = ("random", "gp", "neural")
SELECTION_RULE = "measured feasible quality_goodput descending; config_id, method, run_id ascending"
VALIDATION_ARRIVAL_RATE = .6
VALIDATION_PROTOCOL = {"warmup_s": 0, "ramp_s": 0, "measurement_s": 300,
                       "timeout_s": 180, "drain_s": 180, "max_in_flight": 512}


def read_json(path):
    return json.loads(Path(path).read_text())


def measured_feasible(row, environment_hash):
    metrics = row.get("metrics", {})
    goodput = metrics.get("quality_goodput")
    return (row.get("source") == "measured" and row.get("status") == "ok"
            and row.get("partition") in {"train", "validation"}
            and row.get("workload", {}).get("partition") not in {"test", "holdout"}
            and row.get("environment_hash") == environment_hash
            and metrics.get("feasible") is True
            and isinstance(goodput, (float, int)) and math.isfinite(goodput) and goodput >= 0)


def select_finalist(histories, environment_hash):
    """Selection rule is declared in code before validation, with no test inputs."""
    eligible = []
    for method in METHODS:
        rows = histories.get(method, [])
        if not rows:
            raise ValueError(f"live {method} search has no observations")
        for row in rows:
            if row.get("source") != "measured" or row.get("environment_hash") != environment_hash:
                raise ValueError("search histories must contain measured observations on the frozen environment")
            if measured_feasible(row, environment_hash):
                eligible.append((method, row))
    if not eligible:
        raise ValueError("no measured feasible finalist; no fallback selection after validation")
    method, row = min(eligible, key=lambda item: (-item[1]["metrics"]["quality_goodput"],
                                                item[1]["config_id"], item[0], item[1]["run_id"]))
    return method, copy.deepcopy(row)


def paired_outcomes(rows, question_ids, sampling_seeds):
    """Validate complete provenance and retain every frozen question denominator."""
    expected = dict(zip(question_ids, sampling_seeds))
    lookup = {}
    for row in rows:
        qid = row["question_id"]
        if qid not in expected or qid in lookup or row.get("phase") != "measure":
            raise ValueError("validation scoring must contain exactly one measurement per frozen question")
        if row.get("sampling_seed") not in {None, expected[qid]}:
            raise ValueError("validation response uses an unexpected sampling seed")
        if row.get("status") == "ok" and row.get("sampling_seed") is None:
            raise ValueError("completed validation response lacks sampling seed provenance")
        lookup[qid] = row
    if set(lookup) != set(expected):
        raise ValueError("validation did not record the complete frozen question cohort")
    return [{"question_id": qid, "sampling_seed": expected[qid],
             "correct": bool(lookup[qid].get("correct")) and lookup[qid].get("status") == "ok"
                        and lookup[qid].get("finish_reason") == "stop"}
            for qid in question_ids]


def quality_metrics_pass(row, outcomes, study):
    metrics = row.get("metrics", {})
    accuracy = sum(r["correct"] for r in outcomes) / len(outcomes)
    return (row.get("source") == "measured" and row.get("status") == "ok"
            and metrics.get("feasible") is True and accuracy > 0
            and isinstance(metrics.get("accuracy"), (int, float))
            and math.isclose(metrics["accuracy"], accuracy, abs_tol=1e-10)
            and isinstance(metrics.get("error_rate"), (int, float))
            and 0 <= metrics["error_rate"] <= study["constraints"]["max_error_rate"])


def validation_evidence(row, plan, scored_path, passed):
    return {"source": "measured", "partition": "development_validation",
            "quality_gate_pass": bool(passed), "question_count": len(plan["question_ids"]),
            "question_ids_hash": digest(plan["question_ids"]),
            "sampling_seeds_hash": digest(plan["sampling_seeds_by_question"]),
            "validation_plan_hash": digest(plan), "run_ids": [row["run_id"]],
            "metrics": copy.deepcopy(row["metrics"]),
            "scored_path": str(Path("runs") / row["run_id"] / "scored.jsonl"),
            "scored_sha256": hashlib.sha256(scored_path.read_bytes()).hexdigest()}


def fail_validation(reason, **details):
    status = {"status": "quality_gate_failed", "final_allowed": False,
              "official_test_accessed": False, "reason": reason, **details}
    write_json(continuation.OUT / "validation-status.json", status, immutable=True)
    print("VALIDATION_FAILED", json.dumps(status), flush=True)
    return status


def validate():
    study, environment, capabilities, tokenizer = continuation.context()
    gate_plan = study.get("quality_gate", {})
    if (gate_plan.get("method") != EXACT_NONINFERIORITY_METHOD
            or gate_plan.get("question_count") != 150 or gate_plan.get("confidence") != .95
            or gate_plan.get("margin") != .02 or study["constraints"]["max_accuracy_drop"] != .02
            or study["model"]["max_model_len"] != 8192 or study["decoding"]["max_tokens"] != 4096):
        raise ValueError("continuation requires the preregistered 150-question, 95%, 2pp, 8192-context/4096-output plan")
    criterion = gate_plan.get("execution_criterion", "exact-noninferiority")
    if criterion not in {"exact-noninferiority", "empirical-paired-accuracy"}:
        raise ValueError("unsupported validation execution criterion")
    histories = {method: read_jsonl(continuation.OUT / "search" / method / "observations.jsonl")
                 for method in METHODS}
    selected_method, candidate = select_finalist(histories, environment["environment_hash"])
    baseline = read_json(continuation.OUT / "default-reference.json")
    if not measured_feasible(baseline, environment["environment_hash"]):
        raise ValueError("default-reference.json must contain a measured feasible baseline observation")
    baseline["is_default"] = True
    candidate["is_default"] = False
    questions = select_questions(continuation.DATA, "development_validation")[:150]
    question_ids = [q["question_id"] for q in questions]
    if len(question_ids) != 150 or len(set(question_ids)) != 150:
        raise ValueError("exactly 150 distinct independent development-validation questions required")
    manifest = read_json(continuation.DATA / "train-manifest.json")
    if set(question_ids) & (set(manifest["partitions"]["profiling"]) | set(manifest["partitions"]["calibration"])):
        raise ValueError("independent validation overlaps calibration/profiling")
    seed = study["decoding"]["seed"]
    sampling_seeds = [seed + index for index in range(150)]
    prompts = [prompt_for(q["question"], tokenizer) for q in questions]
    if max(len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts) + 4096 > 8192:
        raise ValueError("development-validation prompt exceeds frozen context; no truncation")
    trace = [{"phase": "measure", "intended_arrival_s": index / VALIDATION_ARRIVAL_RATE}
             for index in range(150)]
    plan = {"partition": "development_validation", "selection_rule": SELECTION_RULE,
            "selected_method": selected_method, "selected_config_id": candidate["config_id"],
            "selected_config": copy.deepcopy(candidate["config"]),
            "default_config": copy.deepcopy(baseline["config"]),
            "selected_source_run_id": candidate["run_id"], "default_source_run_id": baseline["run_id"],
            "candidate_identical_to_default": candidate["config"] == baseline["config"],
            "study_hash": digest(study), "environment_hash": environment["environment_hash"],
            "question_ids": question_ids, "sampling_seeds_by_question": sampling_seeds,
            "question_ids_hash": digest(question_ids), "sampling_seeds_hash": digest(sampling_seeds),
            "prompt_hashes": [digest(p) for p in prompts], "decoding": copy.deepcopy(study["decoding"]),
            "output_token_limit": 4096, "max_model_len": 8192,
            "protocol": copy.deepcopy(VALIDATION_PROTOCOL), "trace": trace,
            "quality_gate": copy.deepcopy(gate_plan), "no_adaptive_extension_or_candidate_substitution": True,
            "arm_order": ["default", "tuned"],
            "source_history_hashes": {method: digest(histories[method]) for method in METHODS}}
    # This immutable artifact precedes either independent model run.
    write_json(continuation.OUT / "independent-validation-plan.json", plan, immutable=True)
    rows, outcomes, evidence = {}, {}, {}
    for arm, selection in (("default", baseline), ("tuned", candidate)):
        workload = {**selection["workload"], "scenario_id": "independent-development-validation",
                    "profile": "short", "partition": "development_validation",
                    "arrival_rate": VALIDATION_ARRIVAL_RATE, "output_token_limit": 4096}
        row = continuation.trial(study, environment, capabilities, tokenizer, selection["config"], workload,
                                 questions, f"independent-{arm}-n150-seed{seed}", default=arm == "default",
                                 phase="validation", protocol=VALIDATION_PROTOCOL, trace=trace,
                                 reference=None if arm == "default" else rows["default"]["metrics"]["accuracy"],
                                 seed=seed)
        rows[arm] = row
        scored_path = continuation.OUT / "runs" / row["run_id"] / "scored.jsonl"
        if row.get("status") != "ok" or not scored_path.exists():
            return fail_validation(f"{arm} independent trial failed or has no scored evidence", observations=rows)
        try:
            outcomes[arm] = paired_outcomes(read_jsonl(scored_path), question_ids, sampling_seeds)
        except ValueError as exc:
            return fail_validation(str(exc), observations=rows)
        usable = quality_metrics_pass(row, outcomes[arm], study)
        evidence[arm] = validation_evidence(row, plan, scored_path, usable)
        if arm == "default" and not usable:
            return fail_validation("independent baseline quality/error constraints failed", observations=rows,
                                   independent_evidence=evidence)
    bootstrap = paired_accuracy_ci(outcomes["default"], outcomes["tuned"], seed=seed)
    exact = paired_accuracy_noninferiority(outcomes["default"], outcomes["tuned"], planned_question_count=150)
    point_pass = (evidence["tuned"]["quality_gate_pass"]
                  and exact["difference_candidate_minus_reference"] >= -.02 - 1e-10)
    execution_pass = point_pass and (criterion == "empirical-paired-accuracy" or exact["quality_gate_pass"])
    claim = ("statistical_noninferiority_established" if exact["quality_gate_pass"]
             else "empirical_pass_statistical_noninferiority_inconclusive" if point_pass
             else "empirical_quality_constraint_failed")
    evidence["tuned"].update(quality_gate_pass=bool(execution_pass), paired_accuracy_ci=bootstrap,
                             paired_noninferiority=exact, quality_claim_status=claim)
    for arm in ("default", "tuned"):
        write_json(continuation.OUT / f"independent-{arm}-outcomes.json", outcomes[arm], immutable=True)
    write_json(continuation.OUT / "independent-validation-evidence.json", evidence, immutable=True)
    if not execution_pass:
        return fail_validation("preregistered independent quality execution gate failed; no test access or candidate substitution",
                               independent_evidence=evidence, selected_method=selected_method)
    baseline["independent_validation"] = evidence["default"]
    candidate["independent_validation"] = evidence["tuned"]
    frozen = freeze_protocol(continuation.OUT / "freeze.json", study, [baseline, candidate], manifest)
    status = {"status": "validated_and_frozen", "final_allowed": True, "official_test_accessed": False,
              "freeze_hash": frozen["freeze_hash"], "quality_claim_status": claim,
              "selected_method": selected_method, "candidate_identical_to_default": plan["candidate_identical_to_default"]}
    write_json(continuation.OUT / "validation-status.json", status, immutable=True)
    print("VALIDATION_FINISHED", json.dumps(status), flush=True)
    return status


def final():
    from vllm_tuner.fixed_compute import run_fixed_compute, validate_frozen_protocol
    freeze_path = continuation.OUT / "freeze.json"
    frozen = validate_frozen_protocol(read_json(freeze_path))
    status = read_json(continuation.OUT / "validation-status.json")
    if status.get("final_allowed") is not True or status.get("freeze_hash") != frozen["freeze_hash"]:
        raise ValueError("final requires the successful independent validation status matching its freeze")
    study, environment, capabilities, tokenizer = continuation.context()
    if digest(study) != digest(frozen["study"]):
        raise ValueError("study changed after freeze; no final protocol adaptation")
    if {row.get("environment_hash") for row in frozen["selected"]} != {environment.get("environment_hash")}:
        raise ValueError("deployment environment changed after validation; no test access")
    if frozen["study"]["final"]["sampling_seeds"] != [724]:
        raise ValueError("continuation final uses exactly the preregistered seed 724")
    # prepare_dataset independently revalidates the freeze before any test fetch.
    prepare_dataset(continuation.DATA, "test", freeze_path=freeze_path)
    arms = run_fixed_compute(freeze_path, data_dir=continuation.DATA, tokenizer=tokenizer,
                             capabilities=capabilities, environment=environment,
                             executable=shutil.which("vllm"), output_root=continuation.OUT / "final",
                             seed=724, ledger_path=continuation.LEDGER, port=8137)
    print("FINAL_FINISHED", json.dumps({"arm_count": len(arms), "freeze_hash": frozen["freeze_hash"]}), flush=True)
    return arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate", "final"))
    args = parser.parse_args()
    if args.stage == "validate":
        result = validate()
        if not result["final_allowed"]:
            raise SystemExit(2)
    else:
        final()


if __name__ == "__main__":
    main()
