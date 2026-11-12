# Tuning vLLM for Small-Model Reasoning Under a Fixed GPU-Time Budget

A completed experiment comparing random search, Gaussian-process optimization and a small neural performance model for serving Qwen3-1.7B on one NVIDIA A10G. The objective was to return more **correct answers on time**, followed by independent quality validation and a held-out test under equal GPU allocations.

**Main finding:** neural and random search tied at the best observed search result. The neural winner came from the common starting configurations; its four learned choices did not improve that result. A larger output allowance improved held-out accuracy in this sample, but the study did not establish a repeatable advantage from the tuned serving configuration.

## Read the study

- [Report in Markdown](REPORT.md)
- [Report as PDF](report.pdf)
- [Editable Word report](report.docx)
- [Documentation index](docs/README.md) and [reproduction guide](docs/reproduction.md)
- [Full evidence release](https://github.com/xushuwei281/vllm-reasoning-study/releases/tag/evidence-2026-11-03) and [download/verification instructions](docs/github-release.md)

The measurements were completed on 3 November 2026; the report was revised for clarity on 4 November. The repository includes implementation, tests, reports, plans, summaries, figures and trained surrogate checkpoints. The release archive supplies the complete raw evidence as well as the original project snapshot.

## Results

The default reference returned 45 correct answers out of 54, with no transport errors. Its quality-adjusted goodput (QAG) was **0.356 correct, timely answers per second**. QAG counts only correct answers completed inside the measurement window while meeting all response-time limits.

Across ten live trials per method, the best observed QAG was **0.433 for both neural and random search**, and **0.400 for the Gaussian process**. Each method received six common starting configurations and four subsequent choices, with separately measured and charged runs. The neural model's four prospective predictions had a mean absolute error of 0.107, versus 0.097 for simply predicting the mean of prior observations.

The selected configuration used a batched-token limit of 1,024 and a sequence limit of 32. Independent validation produced 127/150 correct answers for default and 131/150 for tuned. The predefined empirical accuracy constraint passed; the conservative statistical check did **not** establish population noninferiority within the two-percentage-point margin.

| Held-out configuration | 2,048-token allowance | 4,096-token allowance |
|---|---:|---:|
| Default scheduler limits | 93/128 (72.66%) | 106/128 (82.81%) |
| Tuned: batch 1,024 / sequences 32 | 100/128 (78.12%) | 113/128 (88.28%) |

Each condition received a six-minute allocation of one GPU, including startup, teardown and unused time. All 128 requests completed before their deadlines, so the differences concern correctness rather than additional completions. Tuned minus default was +5.47 percentage points at both allowances, with 95% paired intervals [0.00, 11.72]. Raising the allowance added 13 correct answers under each configuration; the interaction between configuration and allowance was zero. The language-model weights were unchanged.

The entire study used a conservative upper bound of **3.8879 accelerator-hours**, within the original eight-hour budget. The cloud job is confirmed canceled. The estimated cloud charge was **US$5.6625**; the provider omitted its finish timestamp, so accounting conservatively extends through the recorded cancellation observation rather than claiming an exact duration or verified invoice.

## CPU quickstart

Use Python 3.11 and a separate environment. Installation downloads dependencies; the tests use CPU fixtures, small surrogate models and a local fake HTTP server.

```sh
git clone https://github.com/xushuwei281/vllm-reasoning-study.git
cd vllm-reasoning-study
python3.11 -m venv work/analysis-env
work/analysis-env/bin/python -m pip install -e '.[test,report,word]'
work/analysis-env/bin/python -m pytest -q
```

The compact repository passed **262 tests** on 12 November 2026; see the [publication test record](results/github-publication-tests.txt). For the originally tested macOS/arm64 dependency set, see [reproduction.md](docs/reproduction.md).

Optional software demonstrations, using fresh output directories:

```sh
work/analysis-env/bin/python -m vllm_tuner fake-smoke --output work/new-http-smoke
work/analysis-env/bin/python -m vllm_tuner synthetic-demo --output work/new-synthetic-demo --epochs 100
```

Synthetic outcomes exercise the software; they are not measured vLLM performance. These commands do not run the real language model.

To regenerate the editable Word report from the checked-in report text and figures:

```sh
work/analysis-env/bin/python scripts/render_word_report.py
```

This writes `report.docx` with native text, tables, captions and linked references. See [CONTRIBUTING.md](CONTRIBUTING.md) for development and evidence-preservation guidance.

## Reproduce the saved analysis

Use the [full evidence archive and verification steps](docs/github-release.md) to obtain a separate working copy of the original snapshot. Run the analyzer and report builders there. This preserves the publication checkout and supplies the raw request and telemetry records needed for detailed audits.

The Git checkout retains the small evidence files, including the original resource ledger, frozen plans, all method histories, four trained neural checkpoints, validation evidence and final per-question outcomes. Three large raw-run directories are provided in the release archive:

- `results/continuation/runs/`
- `results/continuation/final/runs/`
- `results/measured/runs/`

The archive's SHA-256 and extraction layout are documented in [github-release.md](docs/github-release.md). Model weights, full dataset downloads and software environments are separate dependencies; dataset manifests identify the pinned data used by the study.

For a new CPU-only uncertainty replay, choose a fresh destination instead of overwriting the saved result:

```sh
work/analysis-env/bin/python scripts/ablate_uncertainty.py \
  --observations results/continuation/search/*/observations.jsonl \
  --budget-ledger results/measured/budget.jsonl \
  --study configs/study-cloud.json \
  --output work/uncertainty-replay-new.json
```

The replay uses existing measurements and incurs no new accelerator charge. In the archived replay, all starting designs already contained the best table entry; equal final results therefore did not test whether the uncertainty bonus helps discover good settings. The command refuses an existing output path.

## Project map

| Location | Contents |
|---|---|
| `src/vllm_tuner/` | Serving collection, scoring, resource accounting, search policies and learned surrogate |
| `configs/` | Frozen study settings and development configurations |
| `scripts/` | Experiment orchestration, analysis, report generation and CPU replay |
| `tests/` | Software checks for scoring, isolation, search, recovery and resource controls |
| `results/continuation/` | CUDA campaign evidence, summaries, checkpoints and scientific figures |
| `results/measured/` | Original resource ledger and earlier local-stage evidence |
| `results/synthetic/` | Explicitly synthetic software fixtures, excluded from measured findings |
| `docs/` | Reproduction, methods, experimental decisions and audit records |

The learned component is a three-member, 64/32-hidden-unit PyTorch ensemble. It predicts QAG and feasibility from configuration and workload features. Its selection score combines expected goodput, ensemble disagreement and predicted feasibility. Gradients train this small performance model, not vLLM or Qwen. See [surrogate.md](docs/surrogate.md) and [INTERFACES.md](INTERFACES.md).

## Scope and limitations

- One model, one GPU, one live search per method and 54 measured questions per search trial limit generalization and optimizer comparisons.
- Repeated profiling questions and only 6–9 observations per neural fit limit claims about prediction quality, uncertainty calibration and transfer to new workloads.
- Paired question intervals do not capture variation across repeated deployments. Fixed final-arm order and persistent caches can affect serving behavior.
- These cohorts do not establish reliable P99 guarantees or stability under indefinitely continuing arrivals. Equal GPU allocations do not imply equal token counts or equal computational work.

Historical cloud commands, runtime prerequisites and recovery procedures are documented in [reproduction.md](docs/reproduction.md). They describe the completed experiment; a new live replication requires fresh experiment state and an independently approved resource budget. Preserve the original measured journals and frozen execution source when extending the project.
