# Contribution and review log

This records AI-agent work performed for the project. It does **not** assert student authorship, teammate participation, independent human peer review or instructor approval. Team members must read and understand the implementation and follow their course's attribution rules.

| Role | Actual work in this session | Ownership boundary |
|---|---|---|
| Coordinator/root agent | Read request and inspected workspace/host; integrated package, data and experiment harness; coordinated the only accelerator queue; owned final validation and report claims | CLI, orchestration, workload/data preparation, packaging, README, report, figures/results integration |
| Systems agent | Verified official release/model/backend sources; corrected initial “no installed backend” scope after identifying Apple Metal; installed isolated pinned Metal runtime and downloaded immutable model snapshot; implemented validation, environment inventory and server lifecycle | `config.py`, `environment.py`, `lifecycle.py`, related tests, capability/source notes, defence material |
| ML agent | Supplied the PyTorch ensemble, constrained neural acquisition, random and GP policies, grouped preprocessing/splits, checkpoints and replay support | `surrogate.py`, `search.py`, `replay.py`, related tests and method documentation |
| Evaluation agent | Supplied answer normalization/scoring, request metrics, resource ledger, uncertainty functions and protocol audit | `evaluation.py`, `accounting.py`, `statistics.py`, related tests and evaluation protocol |

## Concrete review records

| Review event | Evidence / disposition |
|---|---|
| Hardware assumption checked | Initial absence of an installed vLLM package was distinguished from unsupported hardware. Official vLLM Metal sources showed this M2 host is eligible. The coordinator requested investigation; isolated installation and CLI preflight passed. |
| Runtime/flag audit | Actual `vllm --version`, `vllm serve --help=all`, and `vllm bench serve --help` were run for 0.32.0+cpu/Metal0.32.0. Every generated server flag was found. Model/kernel success remains a separate live-run check. |
| Offline model-name audit | Coordinator identified that offline snapshot resolution could change the default API model name. Systems command builder now supplies explicit `--served-model-name` using the original model ID; flag presence was verified in installed help. Live retry outcome is recorded by orchestration. |
| Model provenance audit | The Qwen3-1.7B Apache-2.0 card, exact commit and architecture config were retrieved. Weights were cached outside deliverables. No larger tutorial model substituted. |
| Optional-feature audit | Metal source rejects TP>1; no verified local W8A8 activation/kernel path, Ascend/HCCL stack or disaggregation deployment. These are NOT EVALUATED. |
| Lifecycle/constraint tests | Systems agent ran twelve focused tests: invalid configs, revision pinning, default-flag omission, resource charging, startup crash/timeout cleanup and immutable logs all passed before integration. This is software evidence, not serving performance evidence. |
| Run identity and budget audit | Systems review found stale-ID reuse, per-output budget resets and unbounded custom traces. The coordinator added shared CLI ledger and trace validation; systems then strengthened input fingerprints to cover deadline mode, gold/evaluation inputs, capabilities, executable, allocation ledger and hashed subprocess environment. Nonsecret execution controls are saved for reproduction. |
| Final CPU integrity tests | Eighteen added mocked-run tests passed for exact reuse, changed-input rejection, invalid/unbounded schedules, denied-budget behavior, known-invalid outcomes, positive accelerator allocation and client controls. Together with twelve configuration/lifecycle tests, the final focused run was 30 passed. No model or server was launched by these tests. |
| Reproduction command review | Systems checked the README against actual main/search/reproduce-run CLI help and syntax-checked the isolated Metal installer. Installation itself was not rerun during live measurements. The README screen-diagnostic recipe matches its saved ordinary-mode/pending-reference provenance; generic recovery must honor saved execution controls. |
| Agent integration review | Coordinator owns final integrated test results and research claim audit in saved validation outputs. Module-specific passing tests do not imply the complete study passed. |

## Required human review — pending

No human review has been supplied or invented. Before submission, each actual team member should record their own name, date, reviewed files/run IDs, issues found, corrections verified and unresolved limitations. A useful independent review checks:

1. One raw request's final-answer score against the source question without editing the parser after seeing test outcomes.
2. One trial's TTFT/TPOT/SLO arithmetic, timeout treatment and full offered denominator.
3. One reservation's start/stop timestamps, allocated devices and budget charge, including failure/restart time.
4. Train/development/test and repeated-run partition isolation.
5. A plot reproduced from saved records, including its uncertainty and synthetic-versus-measured classification.
6. One optimizer decision reconstructed from its allowed observations and checkpoint.
7. Whether each main report claim is supported, scoped and properly attributed.

Leave an item pending when the reviewer has not completed it. Automated tests and this agent log are preparation for human peer review, not a replacement for the course's peer-review requirement.

## Original local/Metal stage — 3 November 2026, before continuation

This section preserves the completed first-stage record as dated history. Its statement that the live optimizer comparison and held-out experiment were unrun describes that stage, not the later cloud continuation.

The coordinator captured the complete CPU suite: **122 passed in 3.50 seconds** (`results/software-tests.txt`). Three real serial Metal runs used **571.1714282079483 allocated accelerator-seconds** in the shared ledger. Measured-data diagnostic fitting produced a three-member MLP checkpoint with explicit unreliability flags; synthetic replay and HTTP smoke artifacts remain separate. The official test split and final optimizer/fixed-compute experiments remain unrun. Frozen-test download now uses the same strict evidence validator as final execution. The generic reproducer restores saved execution controls where available.

The report PDF was rendered to eight page images and visually inspected for wrapping, table/figure legibility, headers and pagination. No human peer-review completion is asserted.

## Continuation contribution record — 3 November 2026, running snapshot

At approximately 12:38 UTC, the coordinator reported a healthy live cloud campaign with six search trials completed. Independent quality validation, the four-arm held-out test and any optional longer-context diagnostic were still pending. This is a dated progress record; completion and numerical findings must be taken from final exported artifacts and terminal accounting, not inferred from implementation readiness.

| Role | Concrete continuation work | Evidence and current limits |
|---|---|---|
| Coordinator/root agent | Continued bounded baseline development; established the CUDA reference; froze the one-device serving/search plan and coordinated the serial live campaign; preserved the original global accelerator budget; implemented whole-reservation slots, cloud overhead charging, verified export and terminal reconciliation workflow | Frozen study/search plans, raw pilot/reference runs, original ledger, campaign journals and local finish/export coordinator. Final completion remains pending in this snapshot. |
| ML/analysis agent | Audited the three live optimizers, linked-trace data splits and prediction scaling; implemented immutable neural proposal artifacts and pre-outcome diagnostics; preregistered additional paired final contrasts; implemented exact-once method-overhead attribution and a read-only analysis/reporting path | Search policies, learned-proposal artifact code, `analysis-spec.json`, `analyze_continuation.py`, analysis tests and ML review. Live cloud learned artifacts are evidence only when actually exported; no future result is claimed. |
| Evaluation agent | Audited prior incomplete thinking and scoring; implemented the exact paired harmful-discordance bound; separated the unchanged empirical 2pp execution constraint from the 95% statistical noninferiority claim; implemented independent 150-question validation, guarded official-test access and deterministic 128-question held-out selection | `statistics.py`, validator portion of `fixed_compute.py`, `hf_validation.py`, the optional seeded-ID branch in `data.py`, focused tests and `continuation_protocol.md`. Forty-three focused CPU checks passed before the campaign; those checks are software evidence. |
| Evaluation agent | Reviewed final deadlines, seeds, source hashes, full-cohort/censoring behavior, raw artifact locations, export limits and interrupted-run recovery; preregistered an optional independent longer-context diagnostic; implemented local queue/memory stability analysis | `continuation_evaluation_review.md`, `hf_unseen_context.py`, immutable `unseen-context-plan.json`, `analyze_stability.py` and focused tests. Seven optional-diagnostic CPU tests passed; no optional GPU run was launched by this agent. |
| Systems/report agent | Reviewed the running frozen system and built an outcome-aware continuation report path, final presentation/defence material and artifact checks; coordinated inclusion of available stability plots without claiming robust P99 precision | Continuation systems review, report builder and tests. Report publication requires exported measured results and final accounting. |

The following review decisions were made before their relevant held-out outcomes:

1. Retain the original 0.02 absolute accuracy margin. Permit final measurement on independently observed paired feasibility while reporting separately whether the conservative exact 95% bound establishes noninferiority. Do not silently convert an inconclusive bound into a preserved-quality claim.
2. Freeze 150 distinct independent development questions with no adaptive extension or candidate substitution after a failed quality gate. Freeze 128 official-test IDs using shuffle seed 724, with identical question-to-generation-seed mapping across the four arms.
3. Include measured startup, serving, teardown and allocated idle padding in each 360-second final slot; retain actual overshoot and the shorter post-startup service deadline. Count incomplete and unanswered questions in the full cohort.
4. Attribute each method-tagged cloud overhead exactly once during derived analysis, while retaining shared setup separately and preserving raw observations unchanged.
5. Preregister the optional longer-context workload at 12:27:02 UTC, before independent validation or test outcomes. It may run only after the core four-arm experiment and only when the original ledger and remaining lifetime cover both bounded trials plus a five-minute export reserve. Optional outcomes cannot change core choices.
6. Preserve incomplete artifacts and missing measurements explicitly. A missing final aggregate is not permission to fabricate completed arms; partial recovery remains separately labelled. Raw timeout text is never repaired into a correct answer.

The continuation uses a separately authorized, capped cloud allocation managed by the coordinator. This contribution log does not itself authorize resources or attest to a billing invoice. No human peer review, student contribution or unperformed experiment is invented. The pending human-review checklist above remains applicable after the final report is assembled.

## Completed continuation integration

The synchronized continuation contains 30 live optimizer trials, two independent 150-question validation runs, all four 128-question held-out arms and both 54-question longer-context diagnostic arms. All four neural proposal checkpoint manifests and pre-outcome histories were audited; prospective errors and the replay ceiling limitation are recorded in the ML review. The independent evaluation audit recomputed all paired contrasts from full cohorts. The final CPU suite passed 262 tests in 4.88 seconds (14 dependency deprecation warnings). The cloud job is confirmed CANCELED; because the provider omitted its finish timestamp, the original ledger conservatively charges through observed cancellation: 3.8879014056 accelerator-hours total and estimated HF charges at most US$5.6625. These are automated engineering reviews; human peer review remains pending.

The final 13-page report was rendered to page PNGs. Coordinator and evaluation-agent visual checks covered all pages and found no clipping, overlap, missing glyphs or broken tables. The final rendering keeps small tables together and includes a standalone bibliography/artifact map.
