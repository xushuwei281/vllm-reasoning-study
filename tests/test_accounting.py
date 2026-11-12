import pytest
from vllm_tuner.accounting import BudgetExceeded, BudgetLedger, RunJournal


def test_all_reserved_devices_and_failures_are_charged(tmp_path):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", 100, .25)
    assert ledger.begin("run-a", devices=2, max_wall_seconds=20)
    cost = ledger.finish("run-a", wall_seconds=15, optimizer_cpu_seconds=3, status="oom")
    assert cost["accelerator_seconds"] == 30
    assert ledger.summary()["optimizer_cpu_seconds"] == 3
    assert ledger.summary()["remaining_search_accelerator_seconds"] == 45


def test_validation_reserve_and_single_queue(tmp_path):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", 100, .25)
    with pytest.raises(BudgetExceeded):
        ledger.begin("too-large", 1, 76)
    ledger.begin("search", 1, 75)
    with pytest.raises(RuntimeError):
        ledger.begin("concurrent", 1, 1)
    ledger.finish("search", 75)
    with pytest.raises(BudgetExceeded):
        ledger.begin("more-search", 1, 1)
    assert ledger.begin("validation", 1, 25, phase="validation")
    ledger.finish("validation", 25)
    assert ledger.summary()["remaining_accelerator_seconds"] == 0


def test_crash_resume_holds_budget_and_deduplicates(tmp_path):
    path = tmp_path / "costs.jsonl"
    original = BudgetLedger(path, 100, .25)
    original.begin("interrupted", 2, 20)
    resumed = BudgetLedger(path, 100, .25)
    assert resumed.summary()["held_accelerator_seconds"] == 40
    assert resumed.begin("interrupted", 2, 20) is False
    resumed.finish("interrupted", 18)
    resumed.finish("interrupted", 18)
    assert resumed.summary()["finished_runs"] == 1
    assert resumed.summary()["spent_accelerator_seconds"] == 36
    with pytest.raises(ValueError):
        resumed.finish("interrupted", 19)
    with pytest.raises(ValueError):
        BudgetLedger(path, 101, .25)


def test_overrun_cost_is_not_hidden_and_cpu_only_training_is_free_accelerator_time(tmp_path):
    ledger = BudgetLedger(tmp_path / "costs.jsonl", 100, .25)
    ledger.begin("cpu", 0, 5, phase="optimizer")
    ledger.finish("cpu", 7, optimizer_cpu_seconds=6)
    assert ledger.summary()["spent_accelerator_seconds"] == 0
    ledger.begin("overrun", 1, 70)
    assert ledger.finish("overrun", 80)["reservation_overrun"]
    assert ledger.summary()["spent_accelerator_seconds"] == 80
    assert ledger.summary()["overspent"]


def test_results_are_immutable_and_exact_resumes_are_idempotent(tmp_path):
    journal = RunJournal(tmp_path / "observations.jsonl")
    row = {"run_id": "a", "metrics": {"accuracy": 0}}
    assert journal.append(row)
    assert not journal.append(row)
    assert journal.contains("a")
    with pytest.raises(ValueError):
        journal.append({"run_id": "a", "metrics": {"accuracy": 1}})
    assert len(journal.records()) == 1
