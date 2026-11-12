import copy

import numpy as np
import pytest
import torch

from vllm_tuner.surrogate import FeatureEncoder, NeuralSurrogate, grouped_split, masked_loss, targets


def row(i, repetition=0):
    return {
        "run_id": f"run-{i}-{repetition}", "config_id": f"c{i}", "scenario_id": "short",
        "repetition": repetition, "partition": "train", "source": "synthetic",
        "config": {"precision": "bf16", "max_num_batched_tokens": 1024 * (1 + i % 3),
                   "max_num_seqs": 8 * (1 + i // 3), "gpu_memory_utilization": .9,
                   "tensor_parallel_size": 1},
        "workload": {"arrival_rate": .1 + i / 10, "input_tokens_mean": 100,
                     "input_tokens_p95": 200, "output_token_limit": 512,
                     "burstiness": 1, "profile": "short"},
        "metrics": {"quality_goodput": .1 + i / 10, "p99_e2e_s": 5 + i,
                    "feasible": i != 0}, "status": "ok",
    }


def test_groups_keep_repeats_and_trace_together():
    rows = [row(i, j) for i in range(8) for j in range(2)]
    rows[0]["trace_id"] = rows[2]["trace_id"] = "same-trace"
    rows[4]["trace_hash"] = rows[6]["trace_hash"] = "same-trace-hash"
    train, val = grouped_split(rows, 11)
    for key in ("config_id", "run_id", "trace_id", "trace_hash"):
        left = {r[key] for r in train if key in r}
        right = {r[key] for r in val if key in r}
        assert left.isdisjoint(right)


def test_explicit_partition_overlap_and_holdout_rejected():
    records = [row(0), row(0, 1), row(1)]
    records[1]["partition"] = "validation"
    with pytest.raises(ValueError, match="crosses"):
        grouped_split(records)
    records = [row(0), row(1)]
    records[0]["partition"] = "holdout"
    with pytest.raises(ValueError, match="Holdout"):
        grouped_split(records)


def test_preprocessing_only_uses_training_and_allowed_features():
    train, val = row(0), row(1)
    val["workload"]["arrival_rate"] = 1000
    val["workload"]["profile"] = "unseen-mix"
    encoder = FeatureEncoder().fit([train])
    assert encoder.mean[4] == pytest.approx(.1)
    assert encoder.scale[5] == pytest.approx(100)  # constant length uses train magnitude
    a = encoder.transform([val])
    val["metrics"]["quality_goodput"] = 100000
    val["workload"]["actual_output_tokens"] = 900000
    assert np.array_equal(a, encoder.transform([val]))
    assert a[0, -1] == 1  # explicit unseen categorical bucket


def test_zero_goodput_and_censored_latency_mask_do_not_create_nan():
    records = [row(0), row(1)]
    records[0]["metrics"].update(quality_goodput=0, p99_e2e_s=None)
    records[1]["metrics"].update(p99_e2e_s=999, p99_censored=True)
    y, mask, feasible = targets(records)
    assert y[0, 0] == 0 and mask[0, 0] == 1
    assert mask[:, 1].sum() == 0
    predictions = (torch.zeros(2, 2, requires_grad=True), torch.zeros(2, 1, requires_grad=True))
    loss = masked_loss(predictions, y, mask, feasible)
    assert torch.isfinite(loss)
    loss.backward()
    assert predictions[0].grad[:, 1].abs().sum() == 0


def test_actual_training_and_safe_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    records = [row(i) for i in range(12)]
    model = NeuralSurrogate(seed=42, ensemble_size=3, max_epochs=40, patience=10)
    report = model.fit(records)
    assert set(report["training_groups"]).isdisjoint(report["validation_groups"])
    assert report["warning"]
    assert any(m["epochs"][-1]["train_loss"] < m["epochs"][0]["train_loss"] for m in report["members"])
    predictions = model.predict([row(15)])
    assert all(np.isfinite(v) and v >= 0 for v in predictions[0].values())
    checkpoint = tmp_path / "ensemble.pt"
    model.save(checkpoint)
    loaded = NeuralSurrogate.load(checkpoint)
    assert loaded.predict([row(15)]) == predictions


def test_mixed_synthetic_and_measured_forbidden():
    records = [row(0), row(1)]
    records[0]["source"] = "measured"
    with pytest.raises(ValueError, match="mixed"):
        grouped_split(records)


def test_target_scaling_is_masked_and_fitted_to_training_only():
    from vllm_tuner.surrogate import LogTargetScaler
    train=[row(i) for i in range(4)]
    train[0]['metrics']['p99_e2e_s']=None
    y,mask,_=targets(train)
    scaler=LogTargetScaler().fit(y,mask)
    standardized=scaler.transform(y,mask)
    assert torch.allclose(standardized[:,0].mean(),torch.tensor(0.),atol=1e-6)
    assert torch.allclose(standardized[:,0].std(unbiased=False),torch.tensor(1.),atol=1e-6)
    assert standardized[0,1]==0
    valid=mask.bool()
    assert torch.allclose(scaler.inverse(standardized)[valid],y[valid])
    before=scaler.state_dict()
    extreme=torch.tensor([[100.,100.]])
    assert (scaler.transform(extreme,torch.ones_like(extreme))>1).all()
    assert scaler.state_dict()==before


def test_constant_and_missing_target_heads_are_finite():
    from vllm_tuner.surrogate import LogTargetScaler
    y=torch.tensor([[0.,0.],[0.,0.]])
    mask=torch.tensor([[1.,0.],[1.,0.]])
    scaler=LogTargetScaler().fit(y,mask)
    assert scaler.counts==[2,0]
    assert torch.isfinite(scaler.transform(y,mask)).all()
    assert torch.equal(scaler.scale,torch.ones(2))


def test_version_one_checkpoint_keeps_original_prediction_semantics(tmp_path):
    from vllm_tuner.surrogate import PerformanceMLP
    encoder=FeatureEncoder().fit([row(0),row(1)])
    torch.manual_seed(81)
    network=PerformanceMLP(encoder.dimension)
    network.eval()
    state={'format_version':1,'settings':{'seed':81,'ensemble_size':1,'max_epochs':1,'patience':1,'learning_rate':.001,'dropout':.05},
           'encoder':encoder.state_dict(),'models':[network.state_dict()],'report':{'old_checkpoint':True}}
    path=tmp_path/'old.pt';torch.save(state,path)
    loaded=NeuralSurrogate.load(path)
    x=torch.tensor(encoder.transform([row(2)]))
    with torch.no_grad():
        prediction,logits=network(x)
        old_values=torch.expm1(prediction.clamp(0,20)).numpy()[0]
    actual=loaded.predict([row(2)])[0]
    assert actual['goodput_mean']==float(old_values[0])
    assert actual['p99_mean']==float(old_values[1])
    assert actual['feasible_probability']==float(torch.sigmoid(logits)[0,0])
