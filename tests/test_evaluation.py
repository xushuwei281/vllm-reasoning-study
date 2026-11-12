import pytest
from vllm_tuner.evaluation import (normalize_number, parse_numeric_answer, score_answer,
                                   request_metrics, summarize_requests, queue_trace)

SLOS = {"ttft_s": 2, "tpot_s": 1, "e2e_s": 5}


def request(i="a", **updates):
    row = dict(request_id=i, question_id=i, phase="measure", intended_arrival_s=1,
               dispatch_s=1.1, first_token_s=1.3, completion_s=2,
               status="ok", text="#### 42", output_tokens=3, input_tokens=10,
               finish_reason="stop", correct=True, chunks=[{"time_s": 1.3, "token_count": 1},
                                                          {"time_s": 1.6, "token_count": 2}])
    row.update(updates)
    return row


@pytest.mark.parametrize("text,answer", [
    ("Let us compute 999.\n#### -1,234.50", "-2469/2"),
    (r"Final: \boxed{\frac{1}{2}}", "1/2"),
    (r"\boxed{-\frac{1}{2}}", "-1/2"),
    ("<think>#### 10</think>\n#### .5", "1/2"),
    ("#### 2\n#### 2.0", "2"),
])
def test_parser_exact_normalization(text, answer):
    assert parse_numeric_answer(text) == answer


@pytest.mark.parametrize("text", ["The answer is 42", "#### 1,23", "#### 1/0", "#### NaN",
                                 "#### 2 dollars", "#### 1\n#### 2", r"\boxed{1}\boxed{",
                                 "<think>#### 42", "#### 42 3", "#### __import__('os')"])
def test_parser_rejects_ambiguity_or_malformed(text):
    assert parse_numeric_answer(text) is None


def test_timeout_and_truncation_never_correct_even_when_text_contains_gold():
    assert not score_answer("#### 42", "#### 42", status="timeout")["correct"]
    assert not score_answer("#### 42", "42", finish_reason="length")["correct"]
    assert not score_answer("#### 42", "42", finish_reason=None)["correct"]
    assert score_answer(r"\boxed{\frac{1}{2}}", "#### 0.5")["correct"]
    with pytest.raises(ValueError):
        score_answer("#### 42", "invalid gold")


def test_metrics_tpot_chunk_gaps_and_one_token():
    result = request_metrics(request(), SLOS)
    assert result["ttft_s"] == pytest.approx(.2)
    assert result["tpot_s"] == pytest.approx(.35)
    assert result["e2e_s"] == pytest.approx(.9)
    assert result["arrival_e2e_s"] == pytest.approx(1)
    assert result["chunk_gaps_s"] == pytest.approx([.3])
    assert result["single_token_chunk_gaps_s"] == []
    assert not result["itl_complete"]
    one = request_metrics(request(output_tokens=1), SLOS)
    assert one["tpot_s"] is None and one["slo_pass"]


def test_client_queue_cannot_hide_deadline_violation():
    metrics = request_metrics(request(intended_arrival_s=0, dispatch_s=8, first_token_s=8.1, completion_s=9), SLOS)
    assert metrics["e2e_s"] == 1
    assert not metrics["slo_pass"]


def test_measurement_arrival_and_completion_boundaries_are_explicit():
    rows = [request("a"), request("drain", intended_arrival_s=9, dispatch_s=9.1,
                                 first_token_s=9.2, completion_s=10),
            request("outside", intended_arrival_s=10, dispatch_s=10.1, first_token_s=10.2, completion_s=11),
            request("warm", phase="warmup")]
    metrics = summarize_requests(rows, window_start_s=0, window_end_s=10, slos=SLOS,
                                 allocated_accelerators=2, reference_accuracy=1)
    assert metrics["offered"] == 2
    assert metrics["completed"] == 2
    assert metrics["quality_goodput"] == .1
    assert metrics["quality_goodput_per_accelerator"] == .05
    assert metrics["accuracy"] == 1
    assert metrics["deadline_constrained_accuracy"] == .5
    assert metrics["drain_completed"] == 1
    assert metrics["unfinished_at_window_end"] == 1
    assert metrics["tail_sample_warning"]


def test_timeouts_censor_all_offered_tail_and_count_in_accuracy():
    rows = [request(str(i)) for i in range(98)]
    rows += [request("timeout1", status="timeout", completion_s=8),
             request("timeout2", status="timeout", completion_s=8)]
    metrics = summarize_requests(rows, window_start_s=0, window_end_s=10, slos=SLOS,
                                 reference_accuracy=1)
    assert metrics["p99_e2e_s"] is None
    assert metrics["p99_e2e_censored"]
    assert metrics["p99_completed_e2e_s"] == pytest.approx(.9)
    assert metrics["accuracy"] == .98
    assert metrics["error_rate"] == .02
    assert not metrics["feasible"]


def test_tail_rank_with_less_than_one_percent_failures():
    rows = [request(str(i)) for i in range(1000)] + [request("timeout", status="timeout")]
    metrics = summarize_requests(rows, window_start_s=0, window_end_s=10, slos=SLOS,
                                 reference_accuracy=1)
    assert metrics["p99_e2e_s"] == pytest.approx(.9)
    assert not metrics["p99_e2e_censored"]


def test_reference_and_memory_evidence_required():
    pending = summarize_requests([request()], window_start_s=0, window_end_s=10, slos=SLOS)
    assert not pending["feasible"]
    measured = summarize_requests([request()], window_start_s=0, window_end_s=10, slos=SLOS,
                                 reference_accuracy=1, memory_limit_bytes=100,
                                 memory_samples=[{"time_s": 2, "used_bytes": 101}])
    assert not measured["feasible"]


def test_no_negative_times_or_duplicate_request_ids():
    with pytest.raises(ValueError):
        request_metrics(request(first_token_s=.1), SLOS)
    with pytest.raises(ValueError):
        summarize_requests([request(), request()], window_start_s=0, window_end_s=10, slos=SLOS)


def test_queue_accounts_for_never_dispatched_rejection():
    trace = queue_trace([request("a"), request("rejected", dispatch_s=None, first_token_s=None,
                                              completion_s=2, status="rejected")])
    assert max(row["client_outstanding"] for row in trace) == 2
    assert trace[-1]["client_outstanding"] == trace[-1]["client_waiting"] == 0


def test_unknown_token_usage_stays_null_and_fails_tpot_constraint():
    row = request(output_tokens=None)
    metrics = summarize_requests([row], window_start_s=0, window_end_s=10, slos=SLOS,
                                 reference_accuracy=1)
    assert metrics["generated_output_tokens_all"] is None
    assert metrics["known_output_tokens_sum"] == 0
    assert metrics["missing_token_usage_count"] == 1
    assert metrics["quality_goodput"] == 0


def test_thinking_mode_requires_close_when_open_tag_was_in_prompt():
    assert not score_answer(r"Intermediate \boxed{42}", "42", require_thinking_close=True)["correct"]
    assert score_answer(r"Thoughts. </think> \boxed{42}", "42", require_thinking_close=True)["correct"]
