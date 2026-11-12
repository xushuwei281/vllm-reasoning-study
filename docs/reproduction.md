# Reproducing the implementation and measured evidence

The campaign is **COMPLETE** in the audited delivery, and the cloud job is confirmed canceled. These instructions distinguish local software checks, analysis of saved measurements and commands that perform new accelerator work. Machine-readable plans and audited final status determine what actually completed.

The GitHub clone contains code and compact evidence. Download and verify the
[complete evidence release](github-release.md) before running the artifact-only
analysis below. Run that analysis from the separately extracted snapshot, which
contains the raw request, scoring and telemetry files omitted from Git.

## Isolated CPU environment

From the project root, create a fresh Python 3.11 analysis environment. The project pins direct requirements; the macOS/arm64 lock records the originally exercised full environment:

```sh
python3.11 -m venv work/analysis-env
work/analysis-env/bin/python -m pip install -r requirements-lock-macos-cpu.txt
work/analysis-env/bin/python -m pip install --no-deps -e .
work/analysis-env/bin/python -m pip install reportlab==4.4.9
work/analysis-env/bin/python -m pytest -q
```

On another compatible CPU platform, use `pip install -e '.[test,report]'` in its own environment instead of the macOS lock. Installation requires package downloads; tests then use local fixtures, mock processes/HTTP and CPU training. They do not provision a service or run the real language model. Do not install these PyTorch/Transformers pins over a serving runtime.

Focused ML/audit tests:

```sh
work/analysis-env/bin/python -m pytest -q \
  tests/test_surrogate.py tests/test_search.py tests/test_live_analysis.py \
  tests/test_experiment.py tests/test_uncertainty_ablation.py
```

The README's `fake-smoke` and `synthetic-demo` commands need fresh output directories. The latter trains actual neural models on an analytic synthetic table; its serving outcomes remain synthetic.

## Artifact-only analysis

Use an internally consistent local export containing the original ledger, method histories, frozen plans, validation evidence, final-arm outcomes and terminal cloud accounting. A partial download is insufficient for completion.

```sh
work/analysis-env/bin/python scripts/analyze_continuation.py
```

The analyzer preserves raw evidence and writes derived JSON/plots. Inspect `report_status` and `completion_blockers` in `results/continuation/analysis/summary.json`; successful process exit alone is not study completion. Required terminal provenance is `results/continuation/cloud-accounting-final.json`. The provider job must be terminal, its creation-to-terminal allocation reconciled, and the original ledger free of active reservations.

After `report_status` becomes `complete`:

```sh
work/analysis-env/bin/python scripts/build_continuation_report.py
work/analysis-env/bin/python scripts/render_continuation_pdf.py
```

The builder checks completion before replacing `REPORT.md`. The renderer writes `report.pdf`; visually inspect regenerated pages. The older `build_report.py` generates the historical Metal report.

The optional uncertainty-ablation command in the README reads the ledger to establish first-observation chronology but never appends charges. Its source costs are replay bookkeeping. Run it on CPU after releasing the inference allocation to avoid adding cloud idle time.

## Recorded cloud serving stack

The continuation uses `vllm/vllm-openai:v0.10.2` with one 24-GB NVIDIA A10G. Its recorded runtime supplies PyTorch 2.8.0+cu128, Transformers 4.56.1 and huggingface-hub 0.34.4; driver/GPU/runtime inventory is in `results/continuation/environment.json`. Preserve the image's serving stack:

```sh
python3 -m pip install -r requirements-cloud-analysis.txt
python3 -m pip install --no-deps -e .
python3 scripts/hf_continuation.py setup
```

These commands run inside an already allocated runtime and do not create a job. Setup captures environment/capabilities, checks the installed CLI and downloads Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The 0.10.2 preflight uses ordinary `vllm serve --help`; its `--help=all` semantics differ from later releases. Use the image's `python3` interpreter.

Weights, dataset content and caches may require downloads. The frozen GSM8K revision is `3101c7d5072418e28b9008a6636bde82a006892c`. Prepare only training data before validation/freeze:

```sh
python3 -m vllm_tuner prepare-data --directory data --split train
```

The manifest deterministically separates calibration, profiling and development-validation IDs. Gold stays in the evaluator. The final helper fetches official test content only after validating its freeze.

## Live-stage contracts

The scripts use paths relative to their own project root:

| Required input | Role |
|---|---|
| `configs/study-cloud.json` | Frozen model, decoding, grid, SLOs, quality gate and final design |
| `data/train-manifest.json` and training records | Saved partitions, question content and evaluation gold |
| `results/continuation/environment.json`, `capabilities.json` | Actual serving environment checked before reuse |
| `results/measured/budget.jsonl` | Original eight-accelerator-hour ledger, including prior charges |
| `results/continuation/cloud-allocation.json` | Coordinator-recorded allocation creation time and preallocation charges |
| `results/continuation/default-reference.json` | Real feasible default observation used as the development reference |
| `results/continuation/search-plan.json` | Frozen traces/cohort, reference provenance, source hashes, seeds and bounds |
| `results/continuation/analysis-spec.json` | Preregistered final contrasts and accounting interpretation |

Setup and the helpers do not manufacture allocation metadata, reference measurements or frozen search plans. A new independent measurement replication needs fresh immutable experiment state and a newly measured reference on its approved resource, while preserving this archive. Continuing this study retains the original ledger. An archived environment/reference record is not evidence for a replacement device.

Retained pilot interfaces are:

```sh
python3 scripts/hf_continuation.py pilot configs/cloud-pilot-00.json
python3 scripts/hf_continuation.py pilot configs/cloud-default-profile.json
```

These launch real trials. The default specifications omit scheduler limits, and startup logs preserve the effective defaults. The coordinator records the measured reference and freezes the adopted plan before main search. Do not repeat a completed reservation under its old ID or overwrite its evidence.

After prerequisites are satisfied, run stages individually and stop if a predecessor fails:

```sh
python3 scripts/hf_continuation.py campaign
python3 scripts/hf_validation.py validate
python3 scripts/hf_validation.py final
```

`campaign` rotates random/GP/neural by index. Each method independently pays for its common starts and sees only its own history. Actual trace hashes are verified against the ten schedules. Nominal 300-second slots use a 445-second conservative bound plus 30-second accumulated admission margin; each method has a 3,175-second admission ceiling and ten-trial cap. Unique overhead events account for cloud time beyond raw trial reservations.

`validate` freezes the selected measured feasible finalist, measures both arms on the same 150 independent development questions and saves paired quality evidence. An empirical execution pass and statistical noninferiority are separate conclusions. A failed gate returns an error without authorizing test access, candidate substitution or cohort extension.

`final` requires successful validation status and its matching `freeze.json`. It checks the current study/environment before accessing official test data and measures four arms on the entire 128-question cohort. Each targets a 360-second whole-device reservation from startup, including teardown allowance and idle padding. Actual overruns remain charged. The two output allowances are 2,048 and 4,096 with paired question seeds.

The optional context diagnostic has a separate immutable preregistration. After all four core arms exist:

```sh
python3 scripts/hf_unseen_context.py run
```

Its budget/time/export gate may record `not_run`. It uses unused development questions, cannot change core selections and cannot fill missing core evidence.

## Recovery and historical reproduction

Never delete a failed run, reset the ledger or blindly rerun the campaign after interruption. `closed_loop` records pending proposals before dispatch. `experiment.recover_pending` can import exactly one matching completed observation without remeasurement; unfinished allocations first need an auditable terminal charge. See [recovery.md](recovery.md) and [surrogate.md](surrogate.md).

Campaign overhead labels are unique. Reinvoking the complete command after finished rounds is not a general resume mechanism: the coordinator must reconcile pending state and advance the relevant method with its saved policy and trial bound. Setup, fitting, export and idle allocation all remain charged. Final provider termination evidence and local ledger totals are reconciled after the allocation stops; a confirmed terminal-observation upper bound is used when the provider omits its exact finish timestamp.

Original Apple M2/16-GiB shared-memory pilots remain separately versioned. Their runtime was Python 3.12.12, vLLM 0.32.0+cpu with vLLM Metal 0.32.0, MLX 0.32.2 and torch 2.13.0. They do not share training rows or performance curves with CUDA.

`scripts/install_metal.sh` and `scripts/reproduce_saved_pilot.sh` remain available; the latter starts a new charged trial. The explicit interface is:

```sh
work/analysis-env/bin/python -m vllm_tuner reproduce-run --help
```

It requires the original study/run records, exact question/prompt hashes, tokenizer, fresh run ID/output path, compatible environment/capabilities and a declared ledger. A repeat is new evidence subject to host load, thermal state and non-bitwise execution. Exact historical installation/recovery details remain in [capabilities.md](capabilities.md) and [recovery.md](recovery.md).

The delivered terminal record uses the supported conservative observation-bound path: provider state CANCELED is confirmed, while provider finish/duration are explicitly null. Reconciliation checks charged_accelerator_seconds against the UTC creation-to-observation interval. Treat older exact-provider-timestamp recipes above as the path for providers that return those fields. The CPU replay input journal prefix is preserved under results/continuation/analysis-inputs; it is read-only and creates no new spendable allowance.
