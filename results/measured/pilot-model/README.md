# Measured pilot fit: diagnostic only

The three-member PyTorch ensemble was actually trained on measured pilot records. This checkpoint is **not reliable for search, feasibility decisions, latency prediction, or a quality-preservation claim**. It demonstrates that the learning pipeline can consume real measurements while preserving their grouping and censoring.

There are three measured rows but only two independent trace groups. Both low-load allowance runs share the same trace and remain together in training. The earlier default pilot is one internal validation group. No official GSM8K test questions or test outcomes were used. The split was fixed before fitting, and the original observation journal is byte-for-byte unchanged.

| Quantity | Training | Internal validation |
|---|---:|---:|
| Measured run records | 2 | 1 |
| Independent trace groups | 1 | 1 |
| Valid goodput targets | 2 | 1 |
| Uncensored P99 targets | 1 | 0 |
| Positive feasibility labels | 0 | 0 |
| Negative feasibility labels | 2 | 1 |

Each ensemble member has 3,075 parameters. With one independent training group, this model is severely underdetermined. Its goodput absolute error on the single internal validation row was **0.037913 correct SLO-compliant answers/s**, versus **0.005** for the training-mean constant and **0** for an always-zero predictor. These are single-record diagnostics, not statistical estimates of generalization. No confidence interval or neural-method advantage is claimed.

The P99 head saw one training target derived from just two completed requests, and no uncensored validation target. Its extrapolated output is retained under the explicit `unvalidated_prediction` label for inspection; it is not a usable latency forecast. All feasibility labels are negative, so the classifier cannot demonstrate discrimination between feasible and infeasible configurations. Every head is marked unreliable in the checkpoint's own report and in `diagnostic-summary.json`.

The default pilot's omitted scheduler overrides were resolved only in `derived-inputs.jsonl`: batched tokens 2,048 are directly confirmed by the startup log; sequences 256 are inferred from the installed vLLM 0.32.0 initialization code, the Metal platform's out-of-tree enum, balanced performance mode, and the absence of CLI overrides. `default-resolution.json` preserves the exact source excerpts and hashes. The raw launch configuration remains unchanged and is also retained as `original_config` in each derived row.

The saved fit used approximately 0.492 CPU-seconds and reserved no accelerator. Search was not run, no configuration was selected by this checkpoint, and no official final evaluation was performed. Differences among the pilot protocols—including load, deadline, and output allowance—also prevent interpreting this tiny table as a controlled serving-performance comparison.

Artifacts:

- `derived-inputs.jsonl`: three derived feature records, explicit original settings and internal split.
- `default-resolution.json`: installed-source/log evidence for omitted default scheduler settings.
- `diagnostic-ensemble.pt`: actual trained weights, preprocessing, fit history, and reliability limitations.
- `diagnostic-summary.json`: target counts, costs, baselines, model details and limitations.
- `internal-validation-predictions.jsonl`: one internal validation prediction with explicit unreliability labels.
- `provenance.json`: hashes of the source script, input journal, derived rows, evidence and checkpoint.

From the project directory, reproduce into a new directory using the saved evidence:

```bash
python scripts/train_pilot_diagnostic.py \
  --input results/measured/observations.jsonl \
  --output results/measured/pilot-model-reproduced \
  --resolution-evidence results/measured/pilot-model/default-resolution.json
```

The delivered fit was reproduced with identical predictions from that evidence, and the checkpoint was loaded successfully with restricted PyTorch `weights_only=True` deserialization. Runtime measurements can vary. The script refuses to overwrite an existing diagnostic directory or to silently train on additional runs.
