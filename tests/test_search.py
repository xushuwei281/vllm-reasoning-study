import copy

import pytest

from vllm_tuner.search import SearchPolicy, initial_design, measured_incumbent


def row(i):
    return {
        "run_id": f"r{i}", "config_id": f"c{i}", "scenario_id": "s", "partition": "train",
        "source": "synthetic", "is_default": i == 0,
        "config": {"precision": "bf16", "max_num_batched_tokens": 1024 * (i % 3 + 1),
                   "max_num_seqs": 8 * (i // 3 + 1), "gpu_memory_utilization": .9, "tensor_parallel_size": 1},
        "workload": {"arrival_rate": .2, "input_tokens_mean": 100, "input_tokens_p95": 200,
                     "output_token_limit": 512, "burstiness": 1, "profile": "short"},
        "metrics": {"quality_goodput": i / 10, "p99_e2e_s": 10 + i, "feasible": i != 2},
    }


def test_resume_rng_and_selected_candidates(tmp_path):
    candidates = [row(i) for i in range(12)]
    policy = SearchPolicy("random", seed=19)
    a = policy.choose(candidates, [])
    path = tmp_path / "policy.json"
    policy.save(path)
    resumed = SearchPolicy.load(path)
    expected = policy.choose(candidates, [a])
    actual = resumed.choose(candidates, [a])
    assert expected == actual and actual["config_id"] != a["config_id"]


def test_only_provided_observations_affect_gp():
    candidates = [row(i) for i in range(10)]
    altered = copy.deepcopy(candidates)
    for c in altered:
        c["metrics"]["quality_goodput"] = 999999  # hidden table outcomes must be ignored
    p = SearchPolicy("gp", seed=2)
    q = SearchPolicy("gp", seed=2)
    observed = [row(0), row(1), row(2)]
    assert p.choose(candidates, observed)["config_id"] == q.choose(altered, observed)["config_id"]
    assert p.last_decision["observations_seen"] == 3


def test_default_is_one_fixed_control_and_exhausts():
    candidates = [row(i) for i in range(4)]
    policy = SearchPolicy("default")
    assert policy.choose(candidates, [])["config_id"] == "c0"
    assert policy.choose(candidates, []) is None


def test_common_initial_design_independent_of_outcomes():
    candidates = [row(i) for i in range(12)]
    selected = initial_design(candidates, 5, seed=9)
    assert selected[0]["config_id"] == "c0"
    assert len({c["config_id"] for c in selected}) == 5
    for c in candidates:
        c["metrics"] = {}
    assert [c["config_id"] for c in selected] == [c["config_id"] for c in initial_design(candidates, 5, 9)]


def test_infeasible_repetition_disqualifies_incumbent():
    rows = [row(0), row(1), row(2)]
    rows[2]["metrics"]["quality_goodput"] = 999
    assert measured_incumbent(rows)["config_id"] == "c1"
    repeat = copy.deepcopy(rows[1])
    repeat["metrics"]["feasible"] = False
    assert measured_incumbent(rows + [repeat])["config_id"] == "c0"


def test_holdout_cannot_enter_search():
    holdout = row(0)
    holdout["partition"] = "holdout"
    with pytest.raises(ValueError, match="held-out"):
        SearchPolicy("random").choose([row(1)], [holdout])


def test_neural_policy_trains_with_sufficient_paid_observations():
    import torch
    torch.set_num_threads(1)
    candidates = [row(i) for i in range(10)]
    p = SearchPolicy("neural", seed=1, neural_epochs=8)
    chosen = p.choose(candidates, candidates[:6])
    assert chosen["config_id"] not in {r["config_id"] for r in candidates[:6]}
    assert p.last_decision["method_used"] == "neural"
    assert p.surrogate.report["training_records"] + p.surrogate.report["validation_records"] == 6


def test_old_policy_resume_keeps_legacy_surrogate_training(tmp_path):
    import json
    policy=SearchPolicy('neural',seed=9)
    path=tmp_path/'old.json';policy.save(path)
    data=json.loads(path.read_text());data['settings'].pop('neural_target_standardization');path.write_text(json.dumps(data))
    loaded=SearchPolicy.load(path)
    assert loaded.settings['neural_target_standardization'] is False
    assert SearchPolicy('neural').settings['neural_target_standardization'] is True


def test_each_neural_proposal_retains_actual_weights_split_and_diagnostics(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import torch
    from vllm_tuner.surrogate import NeuralSurrogate, observation_key
    torch.set_num_threads(1)
    candidates = [row(i) for i in range(10)]
    observed = copy.deepcopy(candidates[:6])
    observed[-1]['partition'] = 'validation'
    observed[-1]['metrics']['quality_goodput'] = 1000
    observed[-1]['metrics']['p99_censored'] = True
    policy = SearchPolicy('neural', seed=2, neural_epochs=4)
    policy.artifact_directory = tmp_path / 'models'
    before = tmp_path / 'before.json'; policy.save(before)
    chosen = policy.choose(candidates, observed)
    artifact = policy.last_decision['model_artifact']
    report_path = Path(artifact['report'])
    old_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    report = json.loads(report_path.read_text())
    diagnostics = report['diagnostics']
    assert report['provenance']['source'] == 'synthetic'
    assert report['provenance']['training_report']['members'][0]['epochs']
    assert diagnostics['training_only_baselines']['quality_goodput'] == pytest.approx(.2)
    assert diagnostics['splits']['validation']['metrics']['quality_goodput']['count'] == 1
    assert diagnostics['splits']['validation']['metrics']['p99_e2e_s']['count'] == 0
    assert diagnostics['splits']['validation']['metrics']['p99_e2e_s']['mae'] is None
    for identity in ('config_scenario_keys', 'run_ids', 'trace_ids', 'trace_hashes'):
        assert set(diagnostics['splits']['train']['identities'][identity]).isdisjoint(
            diagnostics['splits']['validation']['identities'][identity])
    assert observation_key(chosen) not in diagnostics['splits']['train']['identities']['config_scenario_keys']
    loaded = NeuralSurrogate.load(artifact['checkpoint'])
    assert loaded.predict([chosen])[0] == pytest.approx(policy.last_decision['prediction'], rel=1e-6)
    assert any('early-stopping' in c for c in diagnostics['caveats'])
    assert 'artifact_directory' not in policy.settings

    # A crash before the policy JSON was saved may replay the same deterministic
    # choice. Reuse the verified artifact, never overwrite its learned weights.
    replay = SearchPolicy.load(before); replay.artifact_directory = policy.artifact_directory
    assert replay.choose(candidates, observed) == chosen
    assert replay.last_decision['model_artifact'] == artifact
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == old_hash

    saved = tmp_path / 'policy.json'; policy.save(saved)
    resumed = SearchPolicy.load(saved); resumed.artifact_directory = policy.artifact_directory
    resumed.choose(candidates, observed + [chosen])
    assert resumed.last_decision['model_artifact']['directory'] != artifact['directory']
    assert len(list((tmp_path / 'models').glob('proposal-*/ensemble.pt'))) == 2
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == old_hash

    report_path.write_text('{}')
    replay = SearchPolicy.load(before); replay.artifact_directory = policy.artifact_directory
    with pytest.raises(RuntimeError, match='integrity'):
        replay.choose(candidates, observed)
