"""Frozen reasoning scoring and client-side serving measurements.

No gold answers enter generation. Every offered request remains in denominators.
Empirical quantiles use the nearest-rank convention. Missing/censored values are
JSON null, never a fabricated zero latency.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction
from typing import Any, Mapping, Sequence

_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
_NUMBER_FULL = re.compile(rf"^{_NUMBER}$")
_FRACTION_FULL = re.compile(rf"^({_NUMBER})\s*/\s*({_NUMBER})$")
_LATEX_FRAC_FULL = re.compile(rf"^([+-]?)\\(?:d?frac)\{{({_NUMBER})\}}\{{({_NUMBER})\}}$")


def normalize_number(value: str) -> str | None:
    """Return exact rational form (e.g. 1/2), rejecting malformed/unit strings."""
    value = value.strip().replace("−", "-")
    if len(value) > 256:
        return None
    match = _FRACTION_FULL.fullmatch(value)
    latex = _LATEX_FRAC_FULL.fullmatch(value)
    try:
        if match:
            number = Fraction(match[1].replace(",", "")) / Fraction(match[2].replace(",", ""))
        elif latex:
            sign = -1 if latex[1] == "-" else 1
            number = sign * Fraction(latex[2].replace(",", "")) / Fraction(latex[3].replace(",", ""))
        elif _NUMBER_FULL.fullmatch(value):
            number = Fraction(value.replace(",", ""))
        else:
            return None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return str(number)


def _boxed_values(text: str) -> list[str]:
    """Balanced braces allow \\boxed{\\frac{1}{2}} without an eval parser."""
    result = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    result.append(text[start:index])
                    break
        else:
            result.append("")  # An unfinished final box must not salvage an earlier box.
    return result


def parse_numeric_answer(text: str) -> str | None:
    """Parse explicit final markers only, never the last incidental number.

    Reject unfinished thinking. If thinking closed, only inspect the final
    response. All explicit markers in it must agree; malformed markers reject.
    Accepted: a sole numeric line after ####, or balanced \\boxed{number}.
    """
    if not isinstance(text, str):
        return None
    if "<think>" in text and "</think>" not in text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    values = _boxed_values(text)
    values.extend(re.findall(r"(?m)^\s*####\s*(.*?)\s*$", text))
    if not values:
        return None
    normalized = [normalize_number(value) for value in values]
    if any(value is None for value in normalized) or len(set(normalized)) != 1:
        return None
    return normalized[0]


def score_answer(text: str, gold: str, *, status: str = "ok", finish_reason: str | None = "stop",
                 require_thinking_close: bool = False) -> dict[str, Any]:
    """Gold may be GSM8K's answer text or an already extracted numeric value."""
    expected = parse_numeric_answer(gold)
    if expected is None:
        expected = normalize_number(gold)
    if expected is None:
        raise ValueError("Unparseable gold answer; fix the evaluator input, not the generation")
    predicted = parse_numeric_answer(text)
    if status != "ok":
        reason = status
    elif finish_reason != "stop":
        reason = "truncated" if finish_reason == "length" else "unconfirmed_completion"
    elif require_thinking_close and "</think>" not in text:
        reason = "unfinished_thinking"
    elif predicted is None:
        reason = "invalid_answer"
    else:
        reason = "correct" if predicted == expected else "incorrect_answer"
    valid = (status == "ok" and finish_reason == "stop" and predicted is not None
             and (not require_thinking_close or "</think>" in text))
    return {"correct": bool(valid and predicted == expected), "predicted": predicted,
            "gold": expected, "valid": valid, "reason": reason}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    clean = sorted(float(x) for x in values if _finite(x))
    return clean[max(0, math.ceil(quantile * len(clean)) - 1)] if clean else None


def request_metrics(request: Mapping[str, Any], slos: Mapping[str, float]) -> dict[str, Any]:
    """TTFT/TPOT from client event timestamps; multi-token chunks are not ITL."""
    arrival = request.get("intended_arrival_s")
    dispatch = request.get("dispatch_s")
    first = request.get("first_token_s")
    completion = request.get("completion_s")
    for a, b in [(arrival, dispatch), (dispatch, first), (first, completion), (dispatch, completion)]:
        if _finite(a) and _finite(b) and b < a - 1e-9:
            raise ValueError("Non-monotonic request timestamps")
    ok = request.get("status") == "ok" and _finite(completion) and _finite(dispatch)
    ttft = first - dispatch if _finite(first) and _finite(dispatch) else None
    e2e = completion - dispatch if ok else None
    arrival_e2e = completion - arrival if ok and _finite(arrival) else None
    queue = dispatch - arrival if _finite(dispatch) and _finite(arrival) else None
    n_tokens = request.get("output_tokens")
    tpot = ((completion - first) / (n_tokens - 1)
            if ok and _finite(first) and isinstance(n_tokens, int) and n_tokens >= 2 else None)
    chunks = [chunk for chunk in request.get("chunks", []) if _finite(chunk.get("time_s"))]
    gaps, observed_itl = [], []
    for previous, current in zip(chunks, chunks[1:]):
        gap = current["time_s"] - previous["time_s"]
        if gap < 0:
            raise ValueError("Streaming chunks out of order")
        gaps.append(gap)
        if previous.get("token_count") == 1 and current.get("token_count") == 1:
            observed_itl.append(gap)
    # One-token outputs have undefined TPOT and are explicitly exempt from its SLO.
    tpot_ok = (isinstance(n_tokens, int) and not isinstance(n_tokens, bool) and n_tokens == 1) or (tpot is not None and tpot <= slos["tpot_s"])
    slo_pass = bool(ok and ttft is not None and ttft <= slos["ttft_s"] and tpot_ok
                    and e2e <= slos["e2e_s"]
                    and arrival_e2e is not None and arrival_e2e <= slos["e2e_s"])
    return {"ttft_s": ttft, "tpot_s": tpot, "e2e_s": e2e,
            "arrival_e2e_s": arrival_e2e, "client_queue_s": queue, "slo_pass": slo_pass,
            "chunk_gaps_s": gaps, "single_token_chunk_gaps_s": observed_itl,
            "itl_complete": bool(chunks and all(c.get("token_count") == 1 for c in chunks)),
            "latency_completed": ok}


def summarize_requests(requests: Sequence[Mapping[str, Any]], *, window_start_s: float,
                       window_end_s: float, slos: Mapping[str, float],
                       allocated_accelerators: int = 1, reference_accuracy: float | None = None,
                       max_accuracy_drop: float = 0.02, max_error_rate: float = 0.01,
                       memory_samples: Sequence[Mapping[str, Any]] | None = None,
                       memory_limit_bytes: int | None = None) -> dict[str, Any]:
    """Arrival cohort is [start,end); primary goodput also requires completion < end.

    Drain results contribute to full-cohort answer accuracy and latency, not the
    fixed-window goodput numerator. Service-tail rank assigns failed requests an
    infinite completion latency: null if those failures reach the 99th percentile.
    """
    if window_end_s <= window_start_s or allocated_accelerators < 0:
        raise ValueError("Positive measurement duration and nonnegative allocation required")
    for key in ("ttft_s", "tpot_s", "e2e_s"):
        if not _finite(slos.get(key)) or slos[key] <= 0:
            raise ValueError(f"Positive finite SLO required: {key}")
    ids = [r["request_id"] for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate request_id; raw request journal must be deduplicated")
    cohort = [r for r in requests if r.get("phase") == "measure" and
              _finite(r.get("intended_arrival_s")) and
              window_start_s <= r["intended_arrival_s"] < window_end_s]
    duration = window_end_s - window_start_s
    details = [(r, request_metrics(r, slos)) for r in cohort]
    completed = [(r, m) for r, m in details if m["latency_completed"]]
    in_window = [(r, m) for r, m in completed if r["completion_s"] < window_end_s]
    correct = lambda r: bool(r.get("correct")) and r.get("status") == "ok" and r.get("finish_reason") == "stop"
    good = sum(correct(r) and m["slo_pass"] for r, m in in_window)
    latency_good = sum(m["slo_pass"] for _, m in in_window)
    accuracy = sum(correct(r) for r in cohort) / len(cohort) if cohort else 0.0
    errors = sum(r.get("status") != "ok" for r in cohort)
    error_rate = errors / len(cohort) if cohort else 1.0
    e2es = sorted(m["e2e_s"] for _, m in completed)
    rank = math.ceil(0.99 * len(cohort))
    tail_censored = bool(cohort and len(e2es) < rank)
    p99 = e2es[rank - 1] if rank and not tail_censored else None
    memory_values = [s["used_bytes"] for s in memory_samples or []
                     if _finite(s.get("used_bytes")) and _finite(s.get("time_s")) and
                     window_start_s <= s["time_s"] <= window_end_s]
    peak_memory = max(memory_values) if memory_values else None
    memory_ok = memory_limit_bytes is None or (peak_memory is not None and peak_memory <= memory_limit_bytes)
    accuracy_ok = reference_accuracy is not None and accuracy >= reference_accuracy - max_accuracy_drop
    # Pending quality/memory evidence cannot be interpreted as proof of feasibility.
    feasible = bool(cohort and accuracy_ok and error_rate <= max_error_rate and memory_ok)
    allocation_seconds = allocated_accelerators * duration
    result: dict[str, Any] = {
        "quality_goodput": good / duration, "latency_goodput": latency_good / duration,
        "quality_goodput_per_accelerator": good / allocation_seconds if allocation_seconds else None,
        "correct_per_allocated_accelerator_second": good / allocation_seconds if allocation_seconds else None,
        "accuracy": accuracy, "deadline_constrained_accuracy": sum(correct(r) for r, _ in in_window) / len(cohort) if cohort else 0.0,
        "slo_constrained_accuracy": good / len(cohort) if cohort else 0.0,
        "offered": len(cohort), "completed": len(completed), "completed_in_window": len(in_window),
        "correct_completed_in_window": sum(correct(r) for r, _ in in_window),
        "quality_slo_completed_in_window": good, "error_count": errors, "error_rate": error_rate,
        "status_counts": dict(Counter(r.get("status", "missing") for r in cohort)),
        "truncated_count": sum(r.get("finish_reason") == "length" for r in cohort),
        "slo_violation_count": sum(not m["slo_pass"] for _, m in details),
        "unfinished_at_window_end": sum(not _finite(r.get("completion_s")) or r["completion_s"] >= window_end_s for r in cohort),
        "drain_completed": len(completed) - len(in_window), "measurement_duration_s": duration,
        "drain_duration_s": max([0.0] + [r["completion_s"] - window_end_s for r in cohort if _finite(r.get("completion_s"))]),
        "offered_requests_per_s": len(cohort) / duration,
        "dispatched_requests_per_s": sum(_finite(r.get("dispatch_s")) and r["dispatch_s"] < window_end_s for r in cohort) / duration,
        "completed_requests_per_s": len(in_window) / duration,
        "output_tokens_per_s": sum(r.get("output_tokens") or 0 for r, _ in in_window) / duration,
        "input_tokens_per_s": sum(r.get("input_tokens") or 0 for r, _ in in_window) / duration,
        "generated_output_tokens_all": sum(r["output_tokens"] for r in cohort) if all(r.get("output_tokens") is not None for r in cohort) else None,
        "known_output_tokens_sum": sum(r.get("output_tokens") or 0 for r in cohort),
        "missing_token_usage_count": sum(r.get("output_tokens") is None for r in cohort),
        "reasoning_tokens_all": sum(r["reasoning_tokens"] for r in cohort) if cohort and all(r.get("reasoning_tokens") is not None for r in cohort) else None,
        "final_tokens_all": sum(r["final_tokens"] for r in cohort) if cohort and all(r.get("final_tokens") is not None for r in cohort) else None,
        "p99_e2e_s": p99, "p99_e2e_censored": tail_censored,
        "p99_e2e_population": "all measurement-cohort arrivals; failures occupy infinite tail ranks",
        "tail_sample_warning": len(completed) < 1000,
        "peak_memory_bytes": peak_memory, "feasible": feasible,
        "quality_constraint_status": "pending_reference" if reference_accuracy is None else ("pass" if accuracy_ok else "fail"),
        "memory_constraint_status": "not_configured" if memory_limit_bytes is None else ("pass" if memory_ok else "fail_or_unobserved"),
    }
    for field in ["ttft_s", "tpot_s", "e2e_s", "arrival_e2e_s", "client_queue_s"]:
        values = [m[field] for _, m in completed if m[field] is not None]
        for quantile in (50, 95, 99):
            result[f"p{quantile}_completed_{field}"] = percentile(values, quantile / 100)
    result["p99_client_queue_s"] = percentile([m["client_queue_s"] for _, m in details if m["client_queue_s"] is not None], .99)
    result["stream_chunk_gap_p99_s"] = percentile([x for _, m in details for x in m["chunk_gaps_s"]], .99)
    result["single_token_chunk_gap_p99_s"] = percentile([x for _, m in details for x in m["single_token_chunk_gaps_s"]], .99)
    result["itl_observation_note"] = "Chunk gaps are transport observations; missing per-chunk token counts prohibit complete ITL reconstruction."
    return result


def queue_trace(requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Client outstanding requests and pre-dispatch queue, including timed-out work.

    This is not vLLM's engine waiting queue; scrape that metric separately.
    """
    events = []
    for request in requests:
        arrival, dispatch, end = (request.get(k) for k in ("intended_arrival_s", "dispatch_s", "completion_s"))
        if _finite(arrival):
            events.append((arrival, 0, 1, 1))
        if _finite(dispatch):
            events.append((dispatch, 1, 0, -1))
        if _finite(end):
            events.append((end, 2, -1, -int(not _finite(dispatch))))
    outstanding = waiting = 0
    result = []
    for time_s, _, active_delta, wait_delta in sorted(events):
        outstanding += active_delta
        waiting += wait_delta
        result.append({"time_s": time_s, "client_outstanding": outstanding, "client_waiting": waiting})
    return result
