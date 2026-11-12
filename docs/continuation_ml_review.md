# Continuation ML source audit and contribution log

Review date: 2026-11-03. Prepared by the delegated ML implementation/review agent before the main live comparison. This document records implementation checks and the intended inference scope. Empirical optimizer superiority remains unestablished until measured results are available. The frozen machine-readable campaign plan is authoritative for execution details.

## Protocol reviewed

The proposed static search space contains 12 configurations: batched-token limits 512/1024/2048 crossed with sequence limits 8/16/32/64, GPU memory utilization 0.8, BF16, tensor parallelism 1, chunked prefill enabled and prefix caching disabled. Each optimizer receives six independently measured, identically ordered space-filling starts followed by up to four live adaptive proposals. Histories and paid observations remain separate across random search, constrained GP expected improvement and the neural ensemble.

The profiling workload uses 0.6 requests/second for 90 measurement seconds: 54 measurement requests plus one excluded warmup and one excluded ramp request from the same 56-question profiling cohort. The fixed base generation seed is 724, with the same per-request seed convention across methods. Ten preregistered arrival schedules use bounded ±0.1-second jitter; each trial index receives the same actual schedule across methods and a distinct schedule within each method. Each trial starts a fresh server. The runner rotates method order by trial index.

Seed audit clarification: the two excluded requests are one warmup and one ramp request. The frozen optimizer seed and `trace_seed` metadata are 11. `benchmark.collect` constructs actual payload seeds from `study.decoding.seed + request_index`, so generation uses base 724, including the default/search code paths. An intermediate review confused trace/run metadata with sampling seeds; that warning was withdrawn after tracing request construction. Locally available baseline-pilot requests confirm sampling seeds 724, 725, 726, 727 and 728 for indices 0–4. Request-level `sampling_seed` is authoritative; no execution setting changed.

Nominal trial reservations are 300 seconds. Conservative admission uses `trial_reservation_bound`; actual elapsed costs, failures and overshoot remain charged. A method ceiling equal to exactly `9*300 + bound` needs a preregistered allowance for accumulated slot overshoot to reliably admit trial ten; `max_trials=10` remains the independent trial-count cap. The root coordinator was notified before freeze. Cloud allocation outside trial reservations, including optimizer execution, must be reconciled in the global ledger and kept within the original 28,800 accelerator-second ceiling.

## Changes and checks

| Contribution | Audited behavior | Source |
|---|---|---|
| Train-only target conditioning | Each unmasked `log1p` target is centered/scaled using training rows only. Missing/censored P99 stays masked. Constant and missing heads have finite fallbacks. Checkpoint v2 records the transform; legacy checkpoints and resumed policies retain their prior semantics. | `src/vllm_tuner/surrogate.py`, `search.py` |
| Real neural proposal artifacts | Every guided proposal preserves the actual three-member 64/32 MLP ensemble, exact prior observations, selected prediction, full epoch losses and grouped split diagnostics. Atomic immutable directories and checksums protect earlier models on resume. Artifact paths do not alter policy settings. | `src/vllm_tuner/neural_artifacts.py`, `search.py`; one hookup in `experiment.closed_loop` |
| Live cost comparison | Common-cost checkpoints use only already-completed, already-charged observations. Failures remain charged. Undeclared cross-method run reuse and shared adaptive observations are rejected. Physical unique-run costs and per-method debits are distinct fields. | `src/vllm_tuner/live_analysis.py` |
| Planned-trace verification | Optional expected hashes verify actual schedules by trial index. Duplicate/incomplete planned hashes and repeated within-method trace IDs are rejected. Failed rows without actual hashes remain charged and explicitly unverified. | `validate_planned_trace_hashes`, `audit_planned_traces`, `summarize_live_search` |
| Reproducibility and explanation | Added masking, legacy compatibility, checkpoint prediction, artifact integrity/resume, future-outcome exclusion, cost-checkpoint and planned-trace regression checks. Documented the v2 method and its limitations; preserved the original pilot diagnostic's legacy transform. | `tests/test_surrogate.py`, `test_search.py`, `test_live_analysis.py`, `test_experiment.py`; `docs/surrogate.md`; `scripts/train_pilot_diagnostic.py` |

The neural method uses actual gradient training for up to 200 epochs with early stopping. Its acquisition is feasibility probability times mean goodput plus ensemble standard deviation, with beta 1. This is an optimistic ranking heuristic. Model predictions never replace real benchmark outcomes or independent quality acceptance.

## Inference limits to defend explicitly

The internal split tests **configuration generalization conditional on one fixed profiling question cohort, decoding rule and workload family**. Small arrival-time perturbations create distinct measured schedules; they do not create independent question-content samples. Both `trace_id` and `trace_hash` matter to the connected-group split. Hashes must describe actual schedules, rather than identity-only changes made to bypass grouping.

At 6–9 available observations, the nominal internal split contains only 5–7 fitting groups and 1–2 validation groups. Each ensemble member has approximately 3,075 parameters for the fixed workload's feature encoding. The validation split also selects stopping epochs. Its error, reliability bins and two-standard-deviation coverage are descriptive, optimistic diagnostics; ensemble disagreement is not a calibrated confidence interval. Single-class feasibility labels and sparse uncensored P99 targets further limit what can be learned. Few requests per trial also provide weak empirical P99 tail support.

Four prospective adaptive proposals can establish that the learned policy executed and reveal what it selected under this budget. One search run per method supplies no across-search-seed confidence interval. Independent development finalist validation and the frozen held-out fixed-compute evaluation remain necessary for any final performance or quality claim. No neural win, broad workload transfer, calibrated uncertainty or accuracy noninferiority is claimed in this implementation review.

## Validation and contribution provenance

The implementation tests use fixtures and CPU training; their results are not serving measurements. The ML agent launched no accelerator jobs or paid resources. The root coordinator owns provisioning authorization, live runs, global accounting and final empirical conclusions; a separate evaluation agent owns the held-out evaluation protocol. Agent review is not course-required human peer review.

Final targeted validation: **45 tests passed in 2.68 seconds**, covering `test_surrogate.py`, `test_search.py`, `test_live_analysis.py`, `test_experiment.py` and `test_replay.py`. No core code changed while preparing this review.

Reviewed source SHA-256 values:

```text
surrogate.py        69b04864fb67aaa99e8bf31abab6fa02be0799d1e37957717a42786a7addad89
search.py           b115ec65600d4af1d881f600ced23f804c7db54e4965cc08dfb995e36f65751a
neural_artifacts.py 373f235133b2d3a11b22685cc79760781f91241c587dfa1ad29a08da18dbafbb
live_analysis.py    bbefc4060adfc7b524d472ac3fdba163464d96a91b7efa140e96fb1f60813b57
```

## Measured ML audit after the 30 live search trials

This additional audit uses the synchronized local random/GP/neural histories and four real neural artifacts from the completed search stage. It evaluates the learned optimizer and saved CPU replay; it does not certify held-out completion or terminal cloud accounting.

All **four checkpoints and all 12 manifest payload files** passed SHA-256 verification. Each proposal's `observations.json` exactly equals the corresponding neural-history prefix of 6, 7, 8 or 9 paid observations. These contain only measured development rows from the neural method. The selected future outcome is absent. Run/trace/configuration groups are disjoint across each internal split. Independently recomputed feature normalization and masked target transforms match the training-only checkpoint statistics. Loading each actual checkpoint on CPU reproduces its selected-candidate prediction, with maximum absolute difference **5.96e-8** across all heads. Saved parameter fingerprints, training reports, acquisition values and prospective decision references agree. No future-data leakage was found in these retained inputs or transformations. Checksums establish artifact consistency; they are not an independent timestamp attestation.

The current `surrogate.py`, `search.py`, `neural_artifacts.py` and `live_analysis.py` bytes still match the source hashes frozen in the main search plan.

All 12 ensemble-member training histories end with lower training loss than their first epoch, and the checkpoints retain their selected early-stopping states. Each member has 3,075 parameters. The four fitting/validation record counts are 5/1, 6/1, 6/2 and 7/2. This verifies actual learned-model execution while preserving the small-data limitation.

| Guided live run | Batch / sequences | Predicted QAG | Measured QAG | Absolute prediction error | Prior-visible-mean error |
|---|---|---:|---:|---:|---:|
| `live-neural-06` | 1,024 / 64 | 0.348194 | 0.388889 | 0.040695 | 0.070370 |
| `live-neural-07` | 512 / 32 | 0.346815 | 0.344444 | 0.002370 | 0.015873 |
| `live-neural-08` | 512 / 8 | 0.335767 | 0.111111 | 0.224656 | 0.219444 |
| `live-neural-09` | 1,024 / 16 | 0.229150 | 0.388889 | 0.159739 | 0.082716 |

QAG units are correct, SLO-compliant completions per measurement second. Across these four prospectively selected points, neural QAG MAE is **0.106865**, versus **0.097101** for the mean of all prior visible QAG observations. P99 MAE is **10.567316 seconds**, versus **9.825245 seconds** for the analogous prior-visible P99 mean. The model improves over the QAG mean baseline on the first two selections but is worse overall. This small, adaptively selected set cannot establish a general model ranking; it does not demonstrate predictive advantage here.

Only **one of four** future QAG outcomes and **one of four** future P99 values lie within the respective ensemble mean ± two standard deviations. Those are descriptive coverage counts, not confidence intervals. All 30 live search observations are labeled feasible. Consequently feasibility discrimination is untested: the neural probabilities yield prospective Brier score **0.130947**, while an empirical always-feasible predictor scores zero on these four positive outcomes. That comparison does not validate extrapolation to failures. Internal validation is also selected for early stopping and has only one or two groups; its lower errors cannot establish calibration.

The neural history's maximum QAG, **0.433333**, occurs in `live-neural-05`—the sixth common starting measurement, batch 1,024 / sequences 32. Its best adaptive observation is **0.388889**, so the four guided choices add **zero incumbent improvement**. Random search reaches the same observed maximum in its tenth trial; GP's maximum is **0.400000** from its initial design. These are individual history maxima, not a replicated optimizer ranking. The data support a real trained policy executing four prospective choices, without a demonstrated neural acquisition win.

The verified proposal reports are [proposal 1](../results/continuation/search/neural/models/proposal-001-960be403e45e/report.json), [proposal 2](../results/continuation/search/neural/models/proposal-002-20c62a064d64/report.json), [proposal 3](../results/continuation/search/neural/models/proposal-003-2fae08f0bfed/report.json) and [proposal 4](../results/continuation/search/neural/models/proposal-004-9fedc84b51b4/report.json). Their neighboring manifests retain complete checkpoint/report/observation hashes.

### Actual post-hoc replay and its saturated starting point

The saved [uncertainty ablation](../results/continuation/posthoc-uncertainty-ablation.json) reconstructs all **12/12** configurations from 30 live rows, retaining the first chronological measurement per configuration. Independent reconstruction from the reservation ledger reproduces table SHA-256 `7e580717e68cbd211c003cca7984ba98a550d91255020a07557a2a13619039c6`. The static table contains ten linked trace/configuration groups. Each simulated arm has six paired starts and four genuinely neural choices; its visible-run prefixes, revealed source rows and accumulated replay costs match the saved table. The artifact explicitly records zero new hardware measurements and zero new accelerator charges.

All **24 recorded replay fits** were checked against their exact prior-visible rows: training/validation group keys and recomputed training-only target transforms agree, and each fit contains three member histories with valid selected epochs. This audit reads the completed replay; it does not perform another ablation or select new starts after seeing its outcome.

| Replay seed | Initial best QAG, both betas | Final beta=1 QAG | Final beta=0 QAG | First table-maximum attainment |
|---|---:|---:|---:|---|
| 11 | 0.433333 | 0.433333 | 0.433333 | Common-start trial 5 |
| 29 | 0.433333 | 0.433333 | 0.433333 | Common-start trial 5 |
| 47 | 0.433333 | 0.433333 | 0.433333 | Common-start trial 5 |

**This final-incumbent ablation is uninformative about discovering the optimum because every replay already contains a table maximum before adaptation.** In all six `arms[]` records, `initial_config_ids[4]` is `824c39833bb8`, `initial_source_run_ids[4]` is `live-neural-05`, and `curve[4].best_observed.quality_goodput`, `curve[5].best_observed.quality_goodput` and final `best_observed.quality_goodput` all equal **0.43333333333333335**. A second configuration ties that table maximum. Every `paired_seed_results[].beta1_minus_beta0_goodput` equals zero, but that equality is ceiling-limited and is **not evidence that uncertainty has no value**. The beta arms follow identical adaptive paths for seeds 11 and 29 and different paths for seed 47, demonstrating that the term can alter choices even though this endpoint is saturated.

The replay remains post-hoc simulated search over one noisy, adaptively collected empirical table, with conditional configuration generalization and no new hardware finding. Its three seeds are not independent serving repetitions. The replay artifact SHA-256 checked for this review is `d1939ea2020ab9b51795783009dcbb0e86f2bb5cdfd6d6bc7eba92b13e8a7620`.
