import pytest
from vllm_tuner.statistics import (paired_accuracy_ci, paired_accuracy_noninferiority,
                                   harmful_discordance_upper_bound,
                                   block_bootstrap_mean_ci, fixed_compute_summary)


def test_paired_questions_keep_repeated_seed_clusters():
    left = [{"question_id": q, "sampling_seed": seed, "correct": False}
            for q in range(10) for seed in range(3)]
    right = [dict(row, correct=True) for row in left]
    result = paired_accuracy_ci(left, right, bootstrap_samples=100)
    assert result["question_clusters"] == 10
    assert result["paired_requests"] == 30
    assert result["difference_candidate_minus_reference"] == 1
    assert result["ci"] == [1, 1]
    with pytest.raises(ValueError):
        paired_accuracy_ci(left, right[:-1], bootstrap_samples=100)


def paired_cohort(n, seeds=(1,)):
    return [{"question_id": f"q{i}", "sampling_seed": seed, "correct": i % 2 == 0}
            for i in range(n) for seed in seeds]


def test_exact_zero_harm_gate_requires_enough_independent_questions():
    small = paired_cohort(32)
    descriptive = paired_accuracy_ci(small, small, bootstrap_samples=100)
    assert descriptive["degenerate_bootstrap"] and descriptive["ci"] == [0, 0]
    insufficient = paired_accuracy_noninferiority(small, small, planned_question_count=32)
    assert insufficient["quality_gate_usable"]
    assert not insufficient["quality_gate_pass"]
    assert insufficient["ci"][0] == pytest.approx(-(1 - .05 ** (1 / 32)))

    full = paired_cohort(150)
    result = paired_accuracy_noninferiority(full, full)
    assert result["quality_gate_pass"]
    assert result["harmful_questions"] == 0
    assert result["ci"] == pytest.approx([-.01977343816450829, 1.0])
    assert result["reference_accuracy"] == result["candidate_accuracy"] == .5


def test_exact_gate_rejects_harm_and_optional_sample_size_changes():
    left = paired_cohort(150)
    right = [dict(row) for row in left]
    right[0]["correct"] = False
    result = paired_accuracy_noninferiority(left, right)
    assert result["harmful_questions"] == 1
    assert not result["quality_gate_pass"]
    # Even a statistically sufficient extended cohort cannot pass a fixed-n plan.
    extended = paired_cohort(256)
    changed = paired_accuracy_noninferiority(extended, extended)
    assert not changed["quality_gate_usable"] and not changed["quality_gate_pass"]
    with pytest.raises(ValueError, match="complete question cohort"):
        paired_accuracy_noninferiority(left, right[:-1])


def test_exact_gate_clusters_seed_repetitions_and_checks_pairing():
    left = paired_cohort(32, seeds=tuple(range(10)))
    result = paired_accuracy_noninferiority(left, left, planned_question_count=32)
    assert result["paired_requests"] == 320 and result["question_clusters"] == 32
    assert not result["quality_gate_pass"]
    changed_seed = [dict(row) for row in left]
    changed_seed[0]["sampling_seed"] = 999
    with pytest.raises(ValueError, match="paired within question"):
        paired_accuracy_noninferiority(left, changed_seed, planned_question_count=32)


def test_exact_harm_bound_agrees_with_binomial_tail_and_extreme_cases():
    from scipy.stats import binom
    upper = harmful_discordance_upper_bound(1, 150)
    assert binom.cdf(1, 150, upper) == pytest.approx(.05, abs=1e-12)
    assert harmful_discordance_upper_bound(150, 150) == 1
    assert harmful_discordance_upper_bound(1, 256) < .02
    for harms, n, confidence in [(0, 0, .95), (-1, 150, .95), (151, 150, .95),
                                (True, 150, .95), (0, 150, 1)]:
        with pytest.raises(ValueError):
            harmful_discordance_upper_bound(harms, n, confidence=confidence)


def test_time_block_summary_respects_runs():
    rows = [{"run_id": run, "dispatch_s": i, "e2e_s": 2}
            for run in ("a", "b") for i in range(40)]
    result = block_bootstrap_mean_ci(rows, block_seconds=10, bootstrap_samples=100)
    assert result["independent_runs"] == 2
    assert result["time_blocks"] == 8
    assert result["mean"] == 2 and result["ci"] == [2, 2]


def test_fixed_compute_full_cohort_includes_unoffered_and_late():
    requests = [{"question_id": "a", "status": "ok", "finish_reason": "stop", "correct": True,
                 "completion_s": 2, "output_tokens": 4},
                {"question_id": "b", "status": "ok", "finish_reason": "stop", "correct": True,
                 "completion_s": 12, "output_tokens": 8}]
    result = fixed_compute_summary(requests, cohort_ids=["a", "b", "c"], deadline_s=10,
                                  allocated_accelerators=2, reservation_wall_seconds=15)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["deadline_constrained_accuracy"] == pytest.approx(1 / 3)
    assert result["accelerator_seconds"] == 30
    assert result["generated_tokens"] == 12
    assert result["unoffered_questions"] == 1


def test_fixed_compute_unknown_usage_never_reports_zero_actual_tokens():
    rows = [{"question_id": "a", "status": "timeout", "finish_reason": None,
             "correct": False, "completion_s": 10, "output_tokens": None}]
    result = fixed_compute_summary(rows, cohort_ids=["a"], deadline_s=10,
                                  allocated_accelerators=1, reservation_wall_seconds=12)
    assert result["generated_tokens"] is None
    assert result["known_output_tokens_sum"] == 0
    assert result["missing_token_usage_count"] == 1
