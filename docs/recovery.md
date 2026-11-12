# Installation and interrupted-run recovery

## Local installation

Run `bash scripts/install_metal.sh /absolute/path/to/scratch` from any current directory. The script resolves the project relative to its own location and keeps its bootstrap virtual environment, managed Python 3.12.12, runtime, caches and logs under that scratch directory. It never changes `HOME`, installs system packages, runs `curl | bash`, invokes Homebrew or changes network/security settings.

The dependency freeze contains the exact official vLLM 0.32.0+cpu and Metal 0.32.0 release-wheel URLs and the immutable mlx-lm source revision. The script checks OS/architecture, interpreter version, package versions and CLI help. Installation is separate from model download and accelerator execution. Use a fresh scratch directory when the existing interpreter version does not match; do not delete an environment containing work you still need.

This script was syntax checked after authoring. The equivalent isolated installation commands were executed during this project; the new script itself was not rerun during the live experiment because dependency work could disturb its timings.

## Preserve evidence before restarting

A crash is a feasibility outcome and a budget event, not a reason to erase a run. Preserve the run directory, `plan.json`, offered/request logs, server log, observation journal, budget ledger and optimizer pending state. A partial final line in a journal is corruption requiring investigation; do not silently discard it.

1. Stop the experiment queue. Inspect the specific run's log and owned server PID/process group. Terminate only the process group established for that run. Never use a global `pkill python` or kill an unrelated model service.
2. Inspect the ledger with `BudgetLedger(path, total, validation_fraction).summary()`. An unfinished reservation remains charged at its admitted maximum. This prevents a crash from granting free tuning time.
3. Reconcile actual allocated duration from trustworthy timestamps/reservation records. If stop time cannot be recovered, keep a conservative charge and label the duration uncertain. Never reconstruct allocated time by summing request latencies.
4. Append one terminal budget event with `finish(run_id, wall_seconds, status='interrupted')`, using the original immutable ledger policy. Do not modify or remove its `begin` event. Existing finished records must match exactly; conflicting rewrites are rejected.
5. Preserve the failed run as failed in the result journal. Any post-hoc salvage of complete raw responses must be a separately versioned analysis, not a replacement for the original interrupted outcome.
6. Reconcile `pending.json` against the live result journal before resuming search. If the trial completed, import its existing result and cost rather than running it again. If it did not complete, record the charged failure; a deliberate retry receives a new run ID and a new charged reservation. Keep a copy of the pending decision and its reason for recovery.
7. Resume only after no owned server remains, the ledger has no unresolved active run, and the frozen study/configuration/workload identifiers match. A changed runtime, model, prompt, parser, SLO or output allowance starts a new study version rather than overwriting evidence.

The coordinator must keep one study-level total budget across calibration, all optimizer methods/seeds, failures and final validation. Per-method ledgers and trial counts alone cannot enforce that total. Release a locally allocated accelerator before CPU optimizer work; if an external fleet reservation continues, that reserved interval is still charged.

## Failed startup and memory pressure

A startup failure can arise from API model naming, package compatibility, unsupported backend settings, memory pressure or model loading. Diagnose the exact preserved error. Explicit `--served-model-name` keeps the request's model ID stable when offline loading rewrites the checkpoint to a snapshot path. The launch builder includes this flag.

On this Apple host, configured KV capacity is shared unified memory and can differ from resident use. Leave the Metal pressure probe enabled. Decrease a declared memory/capacity setting only as a newly identified trial; do not alter global OS limits or conceal pressure-related failures. Re-run the correctness smoke before collecting a long sustained-load result.
