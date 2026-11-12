# Learning to Tune vLLM for Compute-Budgeted Small-Model Reasoning

A reproducible coursework research implementation, with a **bounded live Apple Metal feasibility study** and separately labelled CPU software validation. The primary research question is whether a compact neural performance surrogate can find configurations with more correct, SLO-compliant reasoning answers per accelerator-second than a documented default, random search, and Gaussian-process Bayesian optimization.

**Status:** the real server ran. The initial development pilots exposed timeouts and truncated thinking. They do not establish a feasible tuned configuration, a neural optimizer win, or improved intrinsic reasoning. See `REPORT.md` and `results/measured/summary.csv` for the final execution inventory. Synthetic fixtures are not serving results. The official GSM8K test split has not been used.

## Start here

- `REPORT.md` / `report.pdf`: methods, measured findings, evidence limits, next experiment, and references.
- `docs/capabilities.md`: installed-release and backend checks; optional feature status.
- `docs/defence.md`: presentation outline and oral-defence questions.
- `docs/contributions.md`: actual agent contributions; human peer review remains pending.
- `docs/evaluation_protocol.md`, `docs/surrogate.md`: equations, interfaces and design decisions.
- `docs/recovery.md`: interrupted-run and budget reconciliation.
- `results/measured/`: immutable live request streams, server logs, telemetry, observations and cost ledger.
- `results/synthetic/`: explicitly synthetic HTTP fixture, analytic table replay, trained ensemble checkpoints and plots.
- `results/environment/`: sanitized host/runtime manifest, exact CLI help and frozen training-data ID partitions.

## Tested local stack

Apple M2, one 8-core Apple GPU, 8 CPU cores, 16 GiB **shared** memory, macOS 27.0.1. Reference Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, Apache-2.0. The live runtime is Python 3.12.12, vLLM 0.32.0+cpu, vLLM Metal 0.32.0, MLX 0.32.2, torch 2.13.0. The upstream `+cpu` wheel hosts the Metal plugin; the actual model executes on the Apple GPU through MLX. Single-rank Gloo appears in startup logs; NCCL and HCCL are not used.

The analysis/client environment is separate: Python 3.11.4, PyTorch 2.7.1, SciPy 1.16.3, scikit-learn 1.6.1 and Transformers 4.51.3. These are not the serving runtime's dependencies. SciPy 1.15.3 failed to import on this OS; the tested pin is 1.16.3. Do not install the analysis PyTorch pin into the Metal runtime.

## Reproduce CPU checks

Run these commands from the extracted project directory. They create scratch files under `work/`; no global install is needed.

```sh
python3 -m venv work/analysis-env
work/analysis-env/bin/python -m pip install -r requirements-lock-macos-cpu.txt
work/analysis-env/bin/python -m pip install --no-deps -e .
work/analysis-env/bin/python -m pytest -q
work/analysis-env/bin/python -m vllm_tuner fake-smoke --output work/new-http-smoke
work/analysis-env/bin/python -m vllm_tuner synthetic-demo --output work/new-synthetic-demo --epochs 100
```

Use a fresh output directory for each demonstration. Existing immutable outputs are intentionally not overwritten. The analytic fixture is an integration test, not a calibrated simulator of vLLM or evidence that a neural surrogate is better. The supplied CPU lock is for the tested macOS/arm64 environment; `pyproject.toml` pins direct requirements for a fresh compatible platform.

## Reproduce a real run

The included helper installs the pinned Apple runtime without writing into the home directory:

```sh
bash scripts/install_metal.sh "$PWD/work"
work/analysis-env/bin/python -m vllm_tuner preflight \
  --vllm "$PWD/work/metal-env/bin/vllm" --version 0.32.0 \
  --output work/preflight
PYTHONPATH=src work/metal-env/bin/python -m vllm_tuner environment \
  --output work/environment/metal.json
work/analysis-env/bin/python -m vllm_tuner prepare-data --directory work/data
HF_HOME="$PWD/work/cache/huggingface" work/metal-env/bin/python scripts/download_model.py \
  --output work/model-path.txt
```

The download is about 3.8 GB. Model weights, official dataset contents, runtime environments and caches are excluded from the deliverable archive. The frozen dataset revision is `3101c7d5072418e28b9008a6636bde82a006892c`. Only its training split is needed for the supplied pilots.

Reproduce the saved low-load completion diagnostic using its original question IDs, prompt hashes, timing trace and configuration:

```sh
work/analysis-env/bin/python -m vllm_tuner reproduce-run \
  --study configs/screen-metal.json \
  --original-run results/measured/runs/screen-metal-00 \
  --data work/data --tokenizer-path "$(cat work/model-path.txt)" \
  --environment work/environment/metal.json \
  --capabilities work/environment/capabilities.json \
  --vllm "$PWD/work/metal-env/bin/vllm" \
  --hf-home "$PWD/work/cache/huggingface" \
  --ledger work/reproduction/budget.jsonl \
  --output work/reproduction --run-id repeat-screen-00
```

Use the **same ledger path for all configurations, optimizers, seeds, calibration and final validation** in one study. Run one accelerator trial queue. The default ceiling is 28,800 allocated accelerator-seconds, with 25% protected for validation. A ceiling is not permission to report unrun experiments or a promise that repeated 1,000-request tail studies fit this Mac.

The timeout pilot uses `configs/pilot-metal.json` and `results/measured/runs/pilot-metal-default-001`. The output-allowance diagnostic uses `configs/allowance-metal.json` and `results/measured/runs/allowance-metal-2048-00`. Change both `--study` and `--original-run`, and choose a new run ID. Exact reproduction still depends on host memory pressure, concurrent user applications, thermal state and non-bitwise GPU execution; report that noise.

## Run a prospective study after the pilot gates pass

`configs/study.json` is a **provisional** research plan. Confirm that the baseline produces valid reasoning answers and choose feasible loads, output allowances, duration and meaningful quality margins on development data before freezing it. Do not call its planned experiments completed.

1. Read the capability audit and run local CLI preflight. Capture a new sanitized environment manifest using the serving interpreter.
2. Prepare only GSM8K training data. The manifest separates 128 calibration IDs, 512 profiling IDs and 256 development-validation IDs.
3. Establish a default with `run --default`; scheduler flags are omitted, while precision/context/prefix policy/memory headroom remain controlled. Save the resolved scheduler values before encoding a default row for the surrogate.
4. Collect enough independent configuration/scenario runs with valid quality outcomes. Run `search --method random`, `gp`, or `neural` with separate output directories, identical starting designs and the same **global ledger**. Pass a `--max-trial-cost` that covers the printed reservation bound, including startup, both shutdown waits, warmup, ramp and drain. Server restarts are charged.
5. Independently rerun measured incumbents on development-validation questions. Use paired question-level intervals for quality, repeated run/time-block latency analysis, and report uncertainty. A point-estimate screening pass is insufficient for the final quality gate.
6. Freeze the study, two measured independently validated default/finalist selections, parser, token allowances and question IDs with `freeze`. `fixed-compute` requires matched, independently measured development evidence and a usable 95% paired noninferiority interval for the candidate; a bare `quality_gate_pass` boolean is rejected. The exact evidence schema is in `docs/pilot_design.md`.
7. Fetch official test only after the freeze; run the four-arm `fixed-compute` command. It shares a service deadline and full question cohort, records actual whole-reservation accelerator-time, and reports a cost frontier if startup/shutdown costs do not match. Do not describe unmatched totals as equal compute.
8. Run `plot --measured-dir ...` and rebuild report tables from saved observations. Tail claims need adequate completed counts and repeated windows; never erase timeout censoring.

See `python -m vllm_tuner --help` and each subcommand's `--help` for every required argument. There is no hidden paid-cloud provisioning or networking modification.

## Important implementation decisions

| Component | Purpose and defensible boundary |
|---|---|
| `config.py`, `environment.py`, `lifecycle.py` | Validate known constraints; immutable model revisions; version/help checks; own process group; count all reserved-device wall time |
| `data.py` | Stable official dataset line IDs; development partitions; prompts without gold; immutable final freeze |
| `benchmark.py` | Open-loop arrival schedule; no semaphore throttling; explicit client-cap rejection; total-stream deadline; every offered request saved; SSE chunk limits stated |
| `evaluation.py` | Strict numeric final answer, closed thinking block, natural EOS required; TTFT/TPOT/E2E; arrival-cohort denominators; censored all-offered P99 |
| `accounting.py` | Locked append-only budget ledger and run journal; single active reservation; validation reserve; failures/overruns retained |
| `surrogate.py` | Three actual 64/32 PyTorch MLPs; masked log1p Huber regression and feasibility BCE; training-only normalization; whole-group validation |
| `search.py`, `replay.py` | Fixed default, random, constrained GP EI and neural acquisition; same paid observations; no hidden full-table access; resumable policy state |
| `experiment.py` | Serial restart/benchmark loop; input fingerprints; pending proposal reconciliation; conservative admission bounds |
| `fixed_compute.py`, `statistics.py` | Full frozen cohort, two token allowances, actual allocation cost, paired question clusters and hierarchical time blocks |
| `plots.py` | Source-labelled figures and raw-derived tables; missing memory remains missing |

The neural acquisition is `p(feasible) * (mean_goodput + beta * ensemble_std)`, optionally multiplied by an approximate latency-feasibility probability. It ranks untested configurations; only an actual measured, quality-checked run can become a finalist. Ensemble spread is not a calibrated confidence interval. Gradients train the surrogate, not `vllm serve` or sampled answers.

## Limits that must stay visible

- W8A8, TP-versus-replica, disaggregation and HCCL: **NOT EVALUATED** on this single-device Metal setup. No universal quantization or HCCL flag is invented.
- The Mac's configured memory fraction does not measure GPU compute utilization or physical residency. The backend changed effective KV capacity under OS memory pressure; raw startup logs preserve that fact. Device peak-memory telemetry is missing.
- Client-observed first nonempty streaming text is the TTFT observation. Chunks can contain multiple tokens. True within-chunk inter-token times are unavailable; total server token usage, when received, is authoritative.
- Raw partial text after timeout does not establish completed-token throughput. Missing usage stays null.
- A tokenizer-based reasoning/final split is labelled re-encoded text accounting, not exact server-internal token partitioning.
- No measured optimizer superiority, final GSM8K accuracy, hardware portability, memory saving, break-even deployment volume or human peer review is claimed.

## Rebuild the report

```sh
work/analysis-env/bin/python -m pip install reportlab==4.4.9
work/analysis-env/bin/python scripts/build_report.py
work/analysis-env/bin/python scripts/render_report_pdf.py
```

The supplied PDF was rendered and visually checked on all eight pages. To replay the longer saved pilot with an existing scratch installation, `bash scripts/reproduce_saved_pilot.sh /absolute/path/to/scratch` performs preflight and starts a new, separately charged run. This helper was syntax-checked; the equivalent underlying reproduction components were unit-tested.
