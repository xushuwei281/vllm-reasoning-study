"""Question-clustered accuracy intervals and time-block latency summaries."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .evaluation import percentile


EXACT_NONINFERIORITY_METHOD = "exact-paired-harmful-discordance-v1"


def _paired_question_values(reference, candidate):
    def grouped(records):
        groups = defaultdict(list)
        seeds = defaultdict(list)
        for row in records:
            groups[str(row["question_id"])].append(float(bool(row.get("correct", False))))
            seeds[str(row["question_id"])].append(row.get("sampling_seed"))
        return groups, seeds
    left, left_seeds = grouped(reference)
    right, right_seeds = grouped(candidate)
    if not left or set(left) != set(right):
        raise ValueError("Paired intervals require the same complete question cohort")
    for qid in left:
        if len(left[qid]) != len(right[qid]) or sorted(left_seeds[qid], key=str) != sorted(right_seeds[qid], key=str):
            raise ValueError("Repeated seeds/requests must be paired within question")
    questions = sorted(left)
    diffs = [sum(right[q]) / len(right[q]) - sum(left[q]) / len(left[q]) for q in questions]
    return questions, left, right, diffs


def harmful_discordance_upper_bound(harmful_questions: int, question_count: int,
                                   *, confidence: float = .95) -> float:
    """One-sided Clopper-Pearson bound for the probability of question harm.

    A harmful question has a lower candidate mean correctness across the paired
    seeds. Since its paired difference is at least -1, accuracy difference is at
    least minus the probability of harm. This bound is conservative when gains
    offset losses or a harmful question loses less than one accuracy unit.
    """
    if (not isinstance(question_count, int) or isinstance(question_count, bool) or question_count <= 0
            or not isinstance(harmful_questions, int) or isinstance(harmful_questions, bool)
            or not 0 <= harmful_questions <= question_count or not 0 < confidence < 1):
        raise ValueError("Valid binomial counts and confidence in (0,1) required")
    if harmful_questions == question_count:
        return 1.0
    if harmful_questions == 0:
        return -math.expm1(math.log1p(-confidence) / question_count)
    from scipy.special import betaincinv
    return float(betaincinv(harmful_questions + 1, question_count - harmful_questions, confidence))


def paired_accuracy_noninferiority(reference: Sequence[Mapping[str, Any]],
                                  candidate: Sequence[Mapping[str, Any]], *,
                                  planned_question_count: int = 150,
                                  confidence: float = .95, margin: float = .02) -> dict[str, Any]:
    """Exact conservative paired noninferiority gate for a fixed question cohort.

    Question clusters, not seed repetitions, are the independent units. The
    cohort size and confidence must be fixed before validation outcomes. Missing
    requests must be materialized as incorrect before calling. No optional
    sample-size extension or bootstrap degeneracy exception is supported.
    """
    if (not isinstance(planned_question_count, int) or isinstance(planned_question_count, bool)
            or planned_question_count <= 0 or not 0 <= margin < 1):
        raise ValueError("Positive preregistered question count and margin in [0,1) required")
    questions, left, right, diffs = _paired_question_values(reference, candidate)
    harms = sum(value < 0 for value in diffs)
    upper = harmful_discordance_upper_bound(harms, len(questions), confidence=confidence)
    complete = len(questions) == planned_question_count
    return {"method": EXACT_NONINFERIORITY_METHOD,
            "confidence": confidence, "sidedness": "one-sided lower",
            "ci": [-upper, 1.0], "harmful_probability_upper": upper,
            "harmful_questions": harms, "question_clusters": len(questions),
            "paired_requests": len(reference), "planned_question_count": planned_question_count,
            "margin": margin, "quality_gate_usable": complete,
            "quality_gate_pass": complete and upper <= margin,
            "reference_accuracy": sum(sum(left[q]) / len(left[q]) for q in questions) / len(questions),
            "candidate_accuracy": sum(sum(right[q]) / len(right[q]) for q in questions) / len(questions),
            "difference_candidate_minus_reference": sum(diffs) / len(diffs),
            "assumption": "Independent question clusters; equal question weights; fixed sample size before outcomes",
            "warning": None if complete else "Observed question count differs from the preregistered fixed count"}


def _interval(samples: Sequence[float], confidence: float) -> list[float | None]:
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    tail = (1 - confidence) / 2
    return [percentile(samples, tail), percentile(samples, 1 - tail)]


def paired_accuracy_ci(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
                       *, bootstrap_samples: int = 2000, confidence: float = .95,
                       seed: int = 0) -> dict[str, Any]:
    """Candidate-reference accuracy difference with question-cluster bootstrap.

    All repeated sampling seeds for a question move together. Both conditions
    must contain the identical question cohort and repeated-seed multiplicities.
    Missing/unanswered requests must first be materialized with correct=False.
    """
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples required")
    questions, left, right, diffs = _paired_question_values(reference, candidate)
    rng = random.Random(seed)
    samples = [sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(bootstrap_samples)]
    return {"difference_candidate_minus_reference": sum(diffs) / len(diffs),
            "confidence": confidence, "ci": _interval(samples, confidence),
            "reference_accuracy": sum(sum(left[q]) / len(left[q]) for q in questions) / len(questions),
            "candidate_accuracy": sum(sum(right[q]) / len(right[q]) for q in questions) / len(questions),
            "question_clusters": len(questions), "paired_requests": len(reference),
            "bootstrap_samples": bootstrap_samples, "seed": seed,
            "degenerate_bootstrap": len(set(diffs)) == 1,
            "quality_gate_usable": len(questions) >= 30 and len(set(diffs)) > 1,
            "warning": ("Degenerate bootstrap or fewer than 30 question clusters: report descriptive differences; this interval alone does not establish noninferiority"
                        if len(questions) < 30 or len(set(diffs)) == 1 else None),
            "method": "paired question-cluster percentile bootstrap; equal question weights"}


def block_bootstrap_mean_ci(records: Sequence[Mapping[str, Any]], *, value_key: str = "e2e_s",
                            time_key: str = "dispatch_s", block_seconds: float = 10,
                            bootstrap_samples: int = 2000, confidence: float = .95,
                            seed: int = 0) -> dict[str, Any]:
    """Hierarchical run and nonoverlapping time-block bootstrap of mean latency.

    Runs are resampled first; within each sampled run, whole time blocks are
    resampled. Every observation in a block is kept together. This is a mean CI,
    not a P99 CI. Inspect sensitivity to block size and use independent runs.
    """
    if block_seconds <= 0 or bootstrap_samples < 100:
        raise ValueError("Positive blocks and at least 100 bootstrap samples required")
    by_run = defaultdict(list)
    for row in records:
        value, timestamp = row.get(value_key), row.get(time_key)
        if value is not None and timestamp is not None and math.isfinite(value) and math.isfinite(timestamp):
            by_run[str(row["run_id"])].append((float(timestamp), float(value)))
    if not by_run:
        return {"mean": None, "ci": [None, None], "independent_runs": 0,
                "warning": "No uncensored observations; no latency estimate"}
    blocks = {}
    for run, pairs in by_run.items():
        origin = min(timestamp for timestamp, _ in pairs)
        grouped = defaultdict(list)
        for timestamp, value in pairs:
            grouped[int((timestamp - origin) // block_seconds)].append(value)
        blocks[run] = list(grouped.values())
    runs = sorted(blocks)
    rng = random.Random(seed)
    samples = []
    for _ in range(bootstrap_samples):
        run_means = []
        for run in rng.choices(runs, k=len(runs)):
            selected = rng.choices(blocks[run], k=len(blocks[run]))
            flat = [value for block in selected for value in block]
            run_means.append(sum(flat) / len(flat))
        samples.append(sum(run_means) / len(run_means))
    run_means = [sum(value for _, value in by_run[run]) / len(by_run[run]) for run in runs]
    return {"mean": sum(run_means) / len(run_means), "ci": _interval(samples, confidence),
            "confidence": confidence, "independent_runs": len(runs),
            "time_blocks": sum(len(value) for value in blocks.values()),
            "block_seconds": block_seconds, "bootstrap_samples": bootstrap_samples, "seed": seed,
            "warning": "Only one run: within-run uncertainty does not establish deployment reproducibility" if len(runs) < 2 else None,
            "method": "hierarchical run/time-block percentile bootstrap; equal run weights; completed only"}


def fixed_compute_summary(requests: Sequence[Mapping[str, Any]], *, cohort_ids: Sequence[str],
                          deadline_s: float, allocated_accelerators: int,
                          reservation_wall_seconds: float) -> dict[str, Any]:
    """Full frozen cohort denominator, including unoffered/unanswered questions.

    Call once per condition and sampling seed; use paired_accuracy_ci across seeds.
    Deadline is a run-relative cutoff. Report actual total reserved cost, including
    startup, independently of that workload deadline.
    """
    if (not math.isfinite(deadline_s) or not math.isfinite(reservation_wall_seconds)
            or deadline_s <= 0 or reservation_wall_seconds < 0 or allocated_accelerators < 0):
        raise ValueError("Positive deadline and nonnegative reservation duration/allocation required")
    cohort = [str(q) for q in cohort_ids]
    if not cohort or len(set(cohort)) != len(cohort):
        raise ValueError("A nonempty unique frozen question cohort is required")
    lookup = {}
    for row in requests:
        qid = str(row["question_id"])
        if qid not in cohort or qid in lookup:
            raise ValueError("One request per frozen cohort question and seed required")
        lookup[qid] = row
    outcomes = []
    for qid in cohort:
        row = lookup.get(qid, {})
        complete = row.get("status") == "ok" and row.get("finish_reason") == "stop"
        correct = bool(row.get("correct")) and complete
        end = row.get("completion_s")
        timely = correct and end is not None and math.isfinite(end) and end <= deadline_s
        outcomes.append({"question_id": qid, "correct": correct,
                         "deadline_correct": bool(timely), "sampling_seed": row.get("sampling_seed")})
    cost = allocated_accelerators * reservation_wall_seconds
    return {"cohort_size": len(cohort), "offered_questions": len(lookup),
            "unoffered_questions": len(cohort) - len(lookup),
            "accuracy": sum(r["correct"] for r in outcomes) / len(cohort),
            "deadline_constrained_accuracy": sum(r["deadline_correct"] for r in outcomes) / len(cohort),
            "generated_tokens": sum(row["output_tokens"] for row in requests) if all(row.get("output_tokens") is not None for row in requests) else None,
            "known_output_tokens_sum": sum(row.get("output_tokens") or 0 for row in requests),
            "missing_token_usage_count": sum(row.get("output_tokens") is None for row in requests),
            "accelerator_seconds": cost, "reservation_wall_seconds": reservation_wall_seconds,
            "deadline_s": deadline_s,
            "correct_per_accelerator_second": sum(r["deadline_correct"] for r in outcomes) / cost if cost else None,
            "question_outcomes": outcomes}
