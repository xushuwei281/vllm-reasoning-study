# Continuation evaluation review

**Review type:** automated source and experimental-design audit by the Codex evaluation subagent. This is a contribution/review record, not course-required human peer review. No teammate endorsement is implied.

**Initial audit state (historical):** 3 November 2026, approximately 12:07 UTC. The main live campaign was running. Independent validation and official held-out outcomes were not available to this reviewer. The initial sections record implementation checks and interpretation requirements. The later independent outcome audit below records the subsequently downloaded final summaries. No accelerator was launched and no resource ledger was changed by this reviewer.

## Scope and version evidence

Reviewed `scripts/hf_validation.py`, `src/vllm_tuner/data.py`, `fixed_compute.py`, `experiment.py`, `benchmark.py`, `statistics.py`, `evaluation.py`, `lifecycle.py`, `accounting.py`, `live_analysis.py`, the frozen cloud study, and the saved search plan. The SHA-256 values of the seven main evaluation/execution source files matched `results/continuation/search-plan.json` during this audit: validation script, data, fixed-compute runner, experiment queue, statistics, streaming benchmark and process lifecycle. Core code was not edited during the running campaign.

The earlier focused CPU test run passed 43 tests covering paired statistics, final gating, shuffled cohort selection, staged orchestration, incomplete requests and immutable provenance. These tests use synthetic fixtures and do not count as measured model evidence. A passing source audit cannot guarantee runtime completion, correctness of every remote artifact, or experimental benefit.

## Preregistered quality constraint and statistical claim

The allowed accuracy loss remains **0.02 absolute**, and the maximum transport/error rate remains **0.01**. The independent quality cohort contains **exactly 150 distinct questions** from `development_validation`, disjoint from the saved calibration/profiling partitions. The script uses the first 150 rows returned for that saved partition, before observing their generated answers. This is fixed cohort selection, not selection by apparent difficulty.

The protocol deliberately distinguishes two decisions:

| Decision | Preregistered rule | What passing permits |
|---|---|---|
| Execute the held-out measurement | Complete paired evidence; positive measured accuracies; candidate-minus-reference accuracy at least -0.02; error rate at most 0.01; measured feasibility | Run the already specified held-out experiment and report its findings |
| Claim statistical noninferiority | One-sided 95% exact harmful-discordance lower bound at least -0.02 | State statistical noninferiority under the stated independent-question model |

The first rule is explicitly selected by `quality_gate.execution_criterion = "empirical-paired-accuracy"`. It was adopted before the live search/independent-validation outcomes. The user's original brief requires actual development-supported quality constraints and paired confidence intervals; it does not require a 95% noninferiority proof before collecting held-out evidence. This distinction retains the original margin while avoiding a stronger, self-imposed prerequisite being confused with the requested constraint.

For each paired question, let harm mean that the candidate's correctness is lower than the reference's. If repeated seeds are used, aggregate their correctness within each question first. Let `h` be harmful questions among `n` independent question clusters. Candidate-minus-reference population accuracy is at least minus the probability of harm. A one-sided Clopper–Pearson upper bound `U` on harm probability therefore gives the conservative accuracy-difference lower bound `-U`; gains are ignored in that bound. For zero harms, `U = 1 - 0.05**(1/n)`. At `n=150`, the lower bound is approximately **-0.019773438**. One harm can make this conservative proof inconclusive even if an offsetting gain leaves the empirical accuracy difference at zero.

The script preserves the exact method, harm count, confidence, interval and paired accuracies separately from the paired question bootstrap. An empirical execution pass with insufficient exact assurance is labelled **`empirical_pass_statistical_noninferiority_inconclusive`**. It must not be rewritten in the report as proven preserved population quality. The exact method can support **`statistical_noninferiority_established`** only when its own bound passes. Bootstrap degeneracy remains visible; it is not falsely relabelled as independent inferential evidence.

No validation-size extension, candidate substitution or revised margin is allowed after these outcomes. If the empirical execution gate fails, the script saves an explicit failed status and does not create the final freeze or access official test data.

## Independent validation path

The finalist is selected from completed measured random/GP/neural histories using descending measured quality-adjusted goodput, then ascending configuration ID, method and run ID. The default is read from the saved measured reference observation. Selection occurs before the independent validation runs. The source run IDs and history hashes are saved, and selected source observations remain separate from independent validation run IDs.

Before either run, `independent-validation-plan.json` records both configurations, the 150 question IDs, prompt hashes, per-question sampling seeds, decoding, trace, measurement protocol, source hashes and the quality rule. Each arm uses 4,096 output tokens, an 8,192-token model context, the same Qwen thinking-mode prompt and unchanged decoding. Gold answers stay in the evaluator.

Validation arrivals use `i / 0.6` seconds for `i=0...149`, with no warmup/ramp, a 300-second measurement window, 180-second per-request timeout and up to 180 seconds of drain. The last arrival is at approximately 248.33 seconds. This is a **150-question finite cohort at 0.6 requests/second spacing**; because there are no further arrivals during the final approximately 51.67 seconds, its whole-window offered rate is **150/300 = 0.5 requests/second**. It should not be called a full 300-second sustained load at 0.6 requests/second. Its goodput is comparable between its two matched arms, not directly to the 90-second search windows without acknowledging the different boundary protocol.

The implementation checks that every frozen question appears exactly once in scored measurement records. Successful completions must retain the expected sampling seed; duplicates, missing questions, unknown questions and seed mismatches fail the gate. Incorrect, truncated, timed-out and rejected responses do not become correct because a numeric answer appears in partial text. Independent accuracy includes drain completions in the full 150-question denominator; goodput requires completion inside the measurement window and all request-level SLOs.

## Held-out cohort, paired randomness and freeze checks

The final cohort has **128 official GSM8K test questions**. `final.test_cohort_seed=724` shuffles the 1,319 possible official IDs before any test content is read; the first 128 become the fixed cohort. The validator reconstructs this ordering and rejects a mismatch. The cohort seed and model sampling seed happen to share the numeric value 724, but they serve separate purposes.

The final sampling protocol is one replicate with request seeds **724 through 851**, assigned by frozen cohort order. All four arms receive the same question-to-seed mapping. Request seeding is applied to actual server generation; it is not merely an arrival-trace seed. Identical seeds do not promise bit-identical GPU answers under different batching/scheduling, which is why answer quality is measured.

Before `prepare_dataset(..., "test")`, the staged script validates the freeze digest, dataset revision, parser version, two measured finalists, complete independent evidence, matching quality cohort/seeds, unchanged study and deployment environment, and the allowed final seed. Dataset preparation independently checks the freeze again. Test prompts are checked against the frozen context limit; exceeding it aborts rather than silently truncating or choosing different questions. The final runner does not feed any test outcomes back into the candidate, decoder, parser or later arm settings.

## Four-arm resource protocol

The fixed arm order is:

1. Default configuration, 2,048-token allowance.
2. Default configuration, 4,096-token allowance.
3. Tuned configuration, 2,048-token allowance.
4. Tuned configuration, 4,096-token allowance.

All 128 questions are scheduled at time zero for each arm. This is an all-at-once finite-cohort deadline test, **not a sustained-load benchmark**. Sustained-load claims belong to the separate search/validation arrival traces.

Each arm receives a nominal **360-second whole-device reservation on one accelerator**. After server startup, the remaining collection duration is approximately:

`effective_service_seconds = 360 - observed_startup_seconds - 30`

The 30 seconds reserve two possible 15-second teardown waits. There is no final warmup/ramp or post-deadline drain. At the effective deadline, unfinished requests are cancelled and remain in the denominator. After the server is stopped, the serial queue retains the allocated device until the nominal 360-second boundary. Idle padding and parsing time within that reservation are charged. Real overshoot is retained instead of clipped; admission also reserves overhead beyond the nominal slot.

The analysis must show each arm's actual accelerator-seconds, startup seconds, effective service deadline, idle padding and overshoot alongside answer accuracy. The plan's `service_deadline_s=360` is the nominal setting before startup deduction, not 360 seconds of post-readiness generation. These results test **server-start-inclusive deadline capacity with preexisting on-disk model/compiler caches**. They do not isolate steady-state serving speed, and they do not establish an increase in the model's intrinsic reasoning ability.

## Denominators, censoring and uncertainty

The fixed-compute summary materializes every frozen question, including never-dispatched questions after failures. Conventional accuracy requires natural stopping and a correct parsed final answer. Deadline-constrained accuracy additionally requires completion by the effective collection deadline. Both divide by 128. A timeout or length truncation contributes zero correct answers even when partial text contains the gold number.

Actual output tokens are reported only when all relevant server-usage values are available. Otherwise the total is null, with the known-token subtotal and missing-usage count reported separately. Reasoning/final text token subdivisions are re-encoded text diagnostics and must not replace authoritative server totals.

The final runner reports paired question-level bootstrap intervals for tuned-minus-default conventional and deadline-constrained accuracy at each output allowance. The same frozen questions and seeds make these paired comparisons meaningful. The two intervals are pointwise, exploratory intervals; no familywise guarantee across multiple outcomes/allowances is claimed. Any additional comparison between token allowances should likewise retain question pairing and be labelled as an analysis of the frozen four-arm experiment.

For serving latency, failed requests occupy the infinite tail ranks in the all-offered P99 summary. A tail that crosses missing completions is reported censored, not replaced by a low completed-only percentile. Completed-only percentiles remain separately labelled. Cohorts of 54, 128 or 150 responses do not meet the stated 1,000-completion tail target. One final run per arm and one optimizer seed do not establish deployment reproducibility or between-search-seed uncertainty. Time-block intervals cannot manufacture independent deployments from one run.

## Findings that must remain visible in the final analysis

| Finding | Practical consequence and disposition |
|---|---|
| No blocking four-arm control-flow defect found in the inspected normal-success path | Continue the frozen serial campaign; verify actual artifacts before claiming completion. |
| The runtime `live_analysis` helper sums trial reservation costs, while retained-device CPU/idle intervals are charged separately as `cloud-overhead-roundXX-METHOD` | Resolved for final analysis by the separately preregistered `analyze_continuation.py`: it adds each uniquely matched method-tagged overhead to a detached copy, verifies exact-once assignment and keeps shared setup separate. Use these derived curves for allocation claims; do not mutate raw observations or count the derived additions again in the global ledger. |
| Failure-only final deadline fallback uses the configured 360 seconds when a failed observation lacks `measurement_duration_s` | If a final arm fails after creating `execution-protocol.json`, recover its actual startup-deducted cutoff from that file for analysis. Do not present the fallback 360 seconds as the actual service window. Preserve the failed raw record. |
| Fixed arm order, preexisting compiler/model caches and one run per arm | Startup, caching and temporal effects can be confounded with condition. Report actual startup and avoid attributing every difference to scheduler throughput. No post-outcome arm-order change is allowed. |
| The final deadline uses a conservative teardown allowance and measured startup; minor clock/I/O/monitor/shutdown overhead remains | Actual reservations may exceed 360 by a small amount, or more on failure. Report the measured cost frontier if the discrepancy is material. Do not round away an overspend. |
| Validation point feasibility can pass while the exact conservative bound is inconclusive | Collecting held-out evidence is permitted under the explicit empirical rule, but statistical preserved-quality wording remains unsupported. |
| Independent validation is a finite cohort with a quiet final measurement segment | Do not compare its absolute goodput to the main search window as though offered load and boundary handling were identical. |

At final artifact review, reconcile the original global ledger through cloud allocation termination, including remaining idle/setup/CPU intervals, before stating total accelerator-hours. Do not reset the ledger or omit earlier Metal failures. Confirm four final arm records, unchanged freeze hash, the 128-question denominator in every arm, actual sampling seeds, retained failed rows and absence of test-based revisions.

## Export completeness and failure recovery audit

Additional read-only audit completed approximately **12:38 UTC**, while the core campaign remained active. Paths below are relative to the remote control root, normally `/workspace/study`; the project itself is `/workspace/study/project`. No remote operation or frozen-core edit was performed by this audit.

| Remote path | Evidence needed and export treatment |
|---|---|
| `project/results/measured/budget.jsonl` | The single original global resource ledger. Export its latest version, including all begin/finish records; preserve any active reservation instead of assuming zero remaining cost. |
| `project/configs/study-cloud.json`, `project/src/`, `project/scripts/`, dependency files | Exact frozen settings and implementation. Existing local copies can be verified against saved source hashes; do not include model caches or credentials. |
| `project/results/continuation/search-plan.json`, `analysis-spec.json`, `default-reference.json`, `environment.json`, `capabilities.json`, `preflight.json`, `preflight/`, `cloud-allocation.json` | Frozen search/analysis contracts, baseline and runtime/accounting provenance. Preserve failed preflight records and prior allocation metadata where present. |
| `project/results/continuation/observations.jsonl` and `search/{random,gp,neural}/` | Main paid observations, optimizer-specific histories, `policy.json`, `policy.json.pt` where present, contracts, incumbents and any pending/recovery records. Export the complete search tree. |
| `project/results/continuation/search/neural/models/proposal-*/` | Each immutable learned proposal's `ensemble.pt`, `manifest.json`, `observations.json` and `report.json`; these substantiate actual learned weights, training-only data and pre-outcome predictions. |
| `project/results/continuation/independent-validation-plan.json`, `independent-validation-evidence.json`, `independent-default-outcomes.json`, `independent-tuned-outcomes.json`, `validation-status.json`, `freeze.json` | Independent cohort, gate decision, exact/empirical claim status, source-scored hashes and final preregistration. Some files legitimately do not exist when validation fails early; retain that status. |
| `project/results/continuation/runs/` | Main/default/pilot/independent-validation raw runs, plus optional context runs if executed. This is **not** the final experiment's raw directory. |
| `project/results/continuation/final/` | Separate final `observations.jsonl`, `fixed-compute-plan-seed724.json`, aggregate summary/comparisons and the four nested raw run directories. Export this tree independently. |
| `project/data/train.jsonl`, `train-manifest.json`, `test.jsonl`, `test-manifest.json` | Pinned public dataset snapshots/manifests used by the run. Test files appear only after permitted test access. They are needed for exact offline answer rescoring if scored artifacts are incomplete. |
| `project/results/continuation/unseen-context-plan.json`, `unseen-context-execution-plan.json`, `unseen-context-summary.json` | Optional preregistration and status/evidence; raw optional runs remain in the main `runs/` tree. A missing optional summary means not yet attempted, not a measured zero result. |
| `.control/process-NNNN.log` reported by control `/status`, and provider job terminal metadata | Stage errors, timing and terminal state. The exact active log name is supplied by `/status`; do not assume its numeric suffix. Use the control server's redacted log, not authentication/environment dumps. |

The four final raw directories are exactly:

```text
project/results/continuation/final/runs/fixed-default-2048-seed724/
project/results/continuation/final/runs/fixed-default-4096-seed724/
project/results/continuation/final/runs/fixed-tuned-2048-seed724/
project/results/continuation/final/runs/fixed-tuned-4096-seed724/
```

For every run, retain `provenance.json`, `server.log`, `execution-protocol.json`, collector `plan.json`, `offered.jsonl`, `requests.jsonl`, `server-metrics.jsonl`, `telemetry.jsonl`, `collection.json`, `scored.jsonl`, `queue.jsonl` and `observation.json` when they exist. Failure-recovered scoring can additionally appear as `fixed-compute-recovered-scoring.json`. The effective deadline is recorded in `execution-protocol.json`; the authoritative terminal cost is in the global ledger and normally the observation.

The remote control `/archive` endpoint rejects more than **512 MiB of uncompressed input** or 20,000 files. The already exported eight-question pilot alone occupies approximately 6.22 MiB, including 4.69 MiB of server metrics. It is plausible that all search, independent and final raw files together exceed the archive limit even though compressed transfer would be smaller. Export small critical metadata/ledgers first, then the final tree and main run directories in bounded chunks or per run. Do not depend on a single archive of the complete results tree at the last minute. Preserve a final file inventory with byte sizes and SHA-256 values. An archive taken during an active run is a partial, non-atomic snapshot; obtain the final delta after normal completion.

The coordinator subsequently addressed this risk in `work/hf_export_and_sync.py`: after the experiment process is idle, it creates a server-side `evidence-export.zip` of all results trees, dataset manifests and redacted control logs, with a per-file size/SHA-256 manifest. It downloads the resulting compressed file, checks the transfer hash and archive membership, and permits only identical local files or exact append-only JSONL extensions. This avoids the directory endpoint's uncompressed-input ceiling; the resulting compressed download must still fit the file-download limit. The full train/test JSONL corpora are not part of that exporter. They can be regenerated from the pinned public source after permitted test access, and their bytes must match the exported dataset manifest before offline rescoring. The root's local finish coordinator owns export, provider cancellation and final reconciliation; this reviewer did not execute it.

The normal final sequence consumes approximately 24 minutes of nominal reservation across four arms. A six-minute interval without a `TRIAL` print can be normal because output is printed after the charged slot finishes. Monitor the active final run's ledger begin, server readiness/log, raw completion progress and eventual terminal finish. The admission bound is approximately 375 seconds per final arm, but OS/process failure can exceed it; actual cost must remain recorded. A request timing out is an expected measured outcome, not by itself a reason to change the remaining frozen conditions.

Manual remote `/stop` needs special care: it signals the outer campaign process group, whereas `ServerProcess` creates vLLM in a separate session/process group. A terminating SIGTERM can bypass Python cleanup and leave that separately owned server running with an active reservation. The normal runner's cancellation and `server.stop()` path handles its own server; `/stop` is not a substitute for verifying owned-server shutdown. If manual interruption is necessary, the coordinator must verify/stop the owned server group and reconcile its actual allocation, or terminate the provider job after preserving artifacts. Provider expiry terminates the container but can make ephemeral files unavailable; export before expiry.

The aggregate `fixed-compute-summary-seed724.json` and comparisons are written only after the complete four-arm loop. Their absence can therefore coexist with one, two or three fully persisted arm directories. The current analyzer intentionally reads aggregate summaries and marks this state incomplete; it does not silently infer missing arms from raw files. If interruption occurs, preserve all raw directories and the active ledger immediately. Reconstruct any assessable arm **offline in a separately labelled recovery artifact**, using the frozen 128 IDs/seeds, raw/scored responses, actual execution deadline and terminal measured cost. Never write a fabricated core completion summary to enable the optional diagnostic.

`offered.jsonl`, terminal `requests.jsonl` records and server-metric scrapes are flushed incrementally. Telemetry samples are buffered until collection finishes, so a killed run may lack `telemetry.jsonl`; do not invent missing memory samples. A truncated final JSONL line must remain in the preserved original bytes; any complete-prefix recovery belongs in a derived file with the omitted byte range documented. Missing, rejected or unfinished questions remain incorrect in the frozen denominator. Missing token usage remains unknown. Active reservations remain conservatively held until actual shutdown/termination timing is available; no missing finish should be replaced by zero cost.

After the provider job has stopped, the local terminal accounting artifact must reconcile the allocation ID, timezone-aware creation and termination evidence, charged device-seconds, prior charged allocation and the final global ledger total, with no active reservations. Exact provider finish timestamps are used when available. When a terminal CANCELED/ERROR/COMPLETED state is confirmed but the provider omits its finish timestamp, the supported `terminal_observation_upper_bound` path requires explicit unknown actual duration, matching UTC observation/accounting endpoints and a conservative creation-to-observation charge. This is the path used by this delivery. `analyze_continuation.py` requires this reconciliation, all planned search observations with their tagged overhead, and four final arms before it labels the report complete. The terminal artifact can be created locally from verified provider metadata after the final remote export; it need not be recoverable from an already terminated container. Optional context results remain a separately interpreted diagnostic and do not substitute for a missing core arm.

## Evaluation contribution record

An optional unseen-context diagnostic was separately preregistered at **2026-11-03 12:27:02 UTC**, before independent validation or official test outcomes. Its plan hash is `a38bbf5f19054161f187f1734daa8495c974cd9a7fc553d09fb6f3e0e4234e02` in `results/continuation/unseen-context-plan.json`. `scripts/hf_unseen_context.py` reuses the unchanged core-freeze default/tuned configurations on unused development-validation rows 150:206, with 2 warmup/ramp questions and 54 measured questions. All prompts use `long_context_stress`, preserving the reasoning task while adding deterministic irrelevant context. The copied search trace is paired across arms; generation seeds are 13724–13779; main decoding and the 4,096-token allowance remain fixed apart from this declared seed change.

This secondary diagnostic may run only after the core four-arm summary exists and at least **1,190 seconds** remain before the conservative job deadline: two 445-second admission bounds plus a protected 300-second export allowance. The gate also deducts unreconciled cloud allocation from the original ledger's available budget and is checked again before each arm. If admission fails, it records an explicit not-run reason. Eligibility does not inspect core accuracy or goodput, and this optional experiment cannot change the core choices. Its plan contains no measured values. Seven focused CPU tests passed; no optional accelerator experiment was run by this reviewer. Even if executed, the diagnostic compares configurations under a new workload; because both question cohort and context differ from the short-prompt search, it is not a same-question causal estimate of adding context alone.

The evaluation subagent audited prior thinking-loop failures and parser assumptions; recommended a documented development-only repetition repair; derived and implemented the exact harmful-discordance bound; distinguished empirical execution feasibility from statistical noninferiority; implemented staged independent validation and guarded test access; added deterministic held-out cohort selection; and wrote focused CPU tests and protocol documentation. The coordinating agent owns live accelerator execution, budget/paid-resource authorization, final source deployment and interpretation of measured findings. The ML agent's optimizer implementation and audit remain separate contributions. No human peer review or unperformed experiment is claimed here.

The evaluation subagent also added `scripts/analyze_stability.py`, a read-only local analyzer for the saved independent default/tuned request, queue and telemetry records. It produces `analysis/stability-summary.json` and, when trajectories exist, `independent_stability.png`/`.svg`. Client outstanding work and sampled engine waiting are distinguished; whole-device memory is labelled as sampled usage. The diagnostic queue slope uses a uniform one-second grid only through the last offered arrival, avoiding the quiet tail being mistaken for evidence of sustainable service. Slopes are descriptive, not inferential tests. Missing terminal records suppress the slope and flag potential queue undercount; unavailable memory remains null. Five focused CPU tests passed, and a synthetic plotting fixture was visually inspected outside research outputs. The current pre-export local artifact reports `not_available` and contains no invented measurements or plot.

The appropriate final conclusions depend on measured results: a neural optimizer may win, tie or lose; independent quality may pass empirically but remain statistically inconclusive; larger output allowances may add no useful reasoning; and a tuned server may improve cost without improving intrinsic answer quality. Each is a valid outcome when reported with its actual controls, denominators, costs and uncertainty.

## Independent outcome audit after core completion

This later read-only audit examined the downloaded four-arm `final/fixed-compute-summary-seed724.json`, the frozen study/cohort and `analysis/summary.json`. The final summary file SHA-256 was `61e81a0f6e222d19781e79e48dbd2b9ca9805799e1cc73f74318375a880e5c01`; the freeze's internal hash remained `269b08b441e96fc943269bbb8c90c1623f88178788051272d18a4477b6a0a450`. At this audit snapshot, the final raw request directories and terminal cloud accounting had not yet been exported locally. Consequently this audit establishes summary consistency and paired arithmetic, not a new raw-response rescore or a terminal resource reconciliation. The coordinator's subsequent export and accounting audit are separate requirements.

All four conditions have exactly the first 128 IDs from the 1,319-ID order frozen by shuffle seed 724, in the same order. Every condition uses request sampling seeds 724 through 851. The four run IDs are distinct, all execution statuses are `ok`, all 128 questions are offered, none is omitted from the denominator, and no output-token usage is missing. Each saved correctness/deadline-correctness value is boolean. In these outcomes the two indicators agree question by question: every correct answer satisfies its deadline. This does not establish that every incorrect request completed naturally or that configuration differences caused extra completions; those mechanisms require the exported terminal records.

| Condition | Correct / frozen cohort | Effective service deadline, seconds | Actual accelerator-seconds |
|---|---:|---:|---:|
| Default, 2,048 allowance | 93 / 128 | 291.186253 | 360.000101737 |
| Default, 4,096 allowance | 106 / 128 | 291.076734 | 360.000101200 |
| Tuned, 2,048 allowance | 100 / 128 | 293.224139 | 360.000117222 |
| Tuned, 4,096 allowance | 113 / 128 | 293.701370 | 360.000113214 |

The largest difference between actual whole-device reservations is approximately **0.000016022 seconds**. The usable deadlines differ because observed startup is deducted from each otherwise matched 360-second slot. The fixed serial arm order and persistent caches remain possible confounders. Equal total resource allocation does not mean identical usable service windows.

Independent NumPy arithmetic sorted the 128 question IDs, formed the five preregistered within-question contrasts, and generated 2,000 common bootstrap resamples with `default_rng(724)`. It did not call the analyzer's contrast helper. This reproduced every primary point estimate and percentile endpoint exactly; ordinary accuracy yields the same results because it equals deadline accuracy here.

| Primary paired contrast | Difference, percentage points | 95% marginal question-bootstrap interval, points |
|---|---:|---:|
| Tuned minus default, 2,048 | +5.46875 | [0, 11.71875] |
| Tuned minus default, 4,096 | +5.46875 | [0, 11.71875] |
| Larger allowance, default | +10.15625 | [3.90625, 16.40625] |
| Larger allowance, tuned | +10.15625 | [4.68750, 16.40625] |
| Serving-by-allowance interaction | 0 | [-6.25000, 7.03125] |

Each configuration comparison contains 11 correct-answer gains and four losses, for a net gain of seven. Each allowance comparison contains 15 gains and two losses, for a net gain of 13. The one-question difference between the allowance intervals' lower endpoints is reproduced by the common finite bootstrap sample: the discordant questions occupy different positions. It is not an inconsistency. Both configuration-effect intervals include zero. The observations therefore do not establish a repeatable configuration advantage, and the zero interaction does not show that tuning uniquely enabled the larger-allowance benefit. None of these descriptive intervals measures between-deployment or between-optimizer-seed uncertainty.

The independent development gate also retains its original distinction: default accuracy was 127/150 and tuned accuracy 131/150, an empirical difference of +2.66667 points, with paired interval approximately [-1.33333, 6.66667]. Three harmful discordances give an exact conservative lower bound of approximately -5.08771 points. The preregistered empirical 2-point constraint passed, permitting held-out collection; statistical noninferiority remained inconclusive. The held-out results do not retroactively change that validation claim.

The post-hoc uncertainty replay has a further, directly verified ceiling. Reconstructing its first-observation table from `coverage.first_observation_provenance` and the saved live histories matches `table_sha256`. The table's maximum feasible QAG is 0.4333333333. For all six beta/seed arms, the incumbent after six initial observations equals both this maximum and the final incumbent. Each starting design includes configuration `824c39833bb8` from `live-neural-05` at zero-based index 4. All three paired seeds therefore reach the empirical table optimum before adaptation. Equal final beta endpoints are uninformative about discovery of that optimum and are not evidence that uncertainty has no value. The empirical table maximum is not a proven population optimum.

The report builder now derives the held-out interpretation from the five saved primary contrasts, checks those estimates against the four arm summaries, and computes the replay ceiling statement only after verifying the exact source-table hash and initial curves. Focused tests use artificial outcomes, different allowances, a smaller replay table, incomplete evidence and unsaturated equal endpoints to prevent campaign-specific conclusions from being hardcoded. All **45 report-builder tests** passed after these derived-only changes; frozen execution code and raw research evidence were unchanged.

## Final raw-evidence audit after synchronization

After the verified export was synchronized locally, this reviewer inspected `results/continuation/final/observations.jsonl` and all four directories under `results/continuation/final/runs/`. Each journal observation equals its per-run `observation.json`. Every run contains exactly 128 unique offered, terminal and scored records. Raw request fields, text, timings, seeds, usage and finish reasons match the scored copies; correctness was recomputed with the unchanged evaluator against each persisted gold answer, including the required thinking-close and natural-stop checks. Every score object, full-cohort correctness indicator and deadline indicator matches the saved summary. Output-token sums also agree exactly. Effective deadlines match both the raw execution protocol and observations.

| Final condition | Transport status `ok` | Natural stop | Token-limit truncation | Correct | Last terminal response, seconds |
|---|---:|---:|---:|---:|---:|
| Default, 2,048 | 128 | 103 | 25 | 93 | 87.202 |
| Default, 4,096 | 128 | 118 | 10 | 106 | 141.630 |
| Tuned, 2,048 | 128 | 104 | 24 | 100 | 138.459 |
| Tuned, 4,096 | 128 | 118 | 10 | 113 | 196.598 |

All four arms have **zero transport errors, zero missing terminal records and zero requests still running at their effective deadline**. All 128 terminal responses arrive before the respective startup-deducted cutoff. A transport completion includes a response stopped by the token limit; such truncations remain incorrect. The final configuration contrast therefore does **not** reflect additional completed requests or recovery of requests that missed the deadline: every condition finished its entire offered cohort. The observed differences concern answer correctness under different schedules/allowances. The tuned configurations' latest response occurs later in these particular runs, so a general latency-speedup claim would also be unsupported.

The optional longer-context evidence is complete under `results/continuation/runs/unseen-context-{default,tuned}-seed13724/`. Both arms have 56 unique transport completions with no errors: two warmup/ramp requests plus the same 54 measured questions. In each measured cohort, 52 responses stop naturally and two reach the 4,096-token cap. All saved outcome indicators match the raw scored records: both configurations answer **48/54** correctly, while **34/54 default** and **37/54 tuned** correct answers finish before the measurement window closes at elapsed time 100 seconds. QAG is consequently 34/90 = **0.377778** and 37/90 = **0.411111** correct answers per measurement second.

The plan, script and effective-study hashes match their saved preregistration; the sole declared decoding change is seed 13724. Actual per-request seeds are 13724–13779, with measured seeds 13726–13779. Collector prompt hashes match the execution plan in both arms, the copied trace matches the original search trace and its saved hash, and recorded prompt lengths plus the 4,096-token allowance fit the unchanged 8,192-token context. Both configurations and their source selection IDs match the original core freeze. The optional plan's 56 IDs and question-content hashes match development-validation rows 150:206 reconstructed from the local training corpus whose SHA-256 matches the exported manifest. They are disjoint from the initial 150 validation questions, calibration, profiling and official-test cohorts. Optional answer scoring was additionally checked against that pinned training corpus's gold answers.

A separate paired bootstrap recomputation reproduces optional accuracy difference **0.00 points**, interval **[-7.41, 7.41]**, and deadline-accuracy difference **+5.56 points**, interval **[-1.85, 12.96]**. Both intervals include zero. The admission records show outcome-independent time/budget checks passed with approximately 1,919 seconds remaining against a 1,190-second initial requirement, and approximately 1,615 seconds remaining against the 745-second one-arm requirement. The diagnostic therefore ran under its previously registered remaining-resource rule. Its observed QAG difference remains a small, secondary workload observation; it does not change the core selections or establish population generalization.

This raw audit supersedes the earlier summary-only limitation for request completion, truncation and score consistency. It does not certify provider billing timestamps. Terminal resource accounting is separately handled by the coordinator and systems audit; a missing provider finish timestamp must remain explicit rather than being invented.
