# Continuation evaluation protocol and audit

Prepared 3 November 2026 after the user explicitly requested completion of the live optimizer comparison and held-out fixed-compute experiment within the **original** eight-accelerator-hour ceiling. This is an evaluation recommendation and audit record. Concrete settings become preregistered only when the coordinator saves the adopted experiment plan before running those stages; suggestions below are not claims that an experiment ran.

## Existing evidence and budget boundary

The only global resource ledger is `results/measured/budget.jsonl`, with total allocation 28,800 accelerator-seconds and 25% protected for validation. The first three real reservations used 571.1714282079483 seconds, leaving 28,228.82857179205 seconds before continuation. Separate output directories must still use this original ledger; no new ledger may reset the ceiling. The initial continuation request prohibited paid provisioning. Any later explicit user authorization for a service is separate from this scientific protocol and does not reset the accelerator-hour ceiling.

The original default run offered four measurement requests and all timed out. The single-sequence 1,024-token diagnostic completed two transport streams but truncated both. At 2,048 tokens, one answer completed correctly at 1,478 tokens while the second timed out after producing approximately 1,885 re-encoded text tokens. Its partial final answer was not counted as completion. These observations establish repetition/allowance/queueing concerns, not proof that the model cannot answer GSM8K.

## Developmental decoding repair, before serving selection

The Qwen model card explicitly recommends a presence penalty of 1.5 when significant endless repetition occurs. It gives thinking-mode sampling temperature 0.6, top-p 0.95, top-k 20 and min-p 0; presence penalties can also reduce quality or mix languages, so the change requires development answer scoring. A bounded first pilot should retain thinking mode and test the documented penalty with enough output allowance (for example 4,096 tokens and a context limit that actually accommodates it). [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B).

If thinking remains impractical under the fixed local resource budget, a separately declared non-thinking protocol is supported by the same checkpoint. Its documented sampling settings are temperature 0.7, top-p 0.8, top-k 20 and min-p 0. The prompt must still require explicit mathematical reasoning and a boxed final answer. This would be a declared development-selected inference mode, not hidden answer shortening or a serving-side speedup attributed to scheduler settings. The mode, prompt, sampling and output allowance must then remain identical across default/random/GP/neural serving comparisons. Any claim compares the chosen protocol only; it does not imply equivalence to the prior thinking-mode baseline.

The existing `evaluate_collected` hardcodes `require_thinking_close=True`. That is correct for the prior thinking trials but would wrongly reject naturally completed non-thinking answers. Make the requirement an explicit frozen protocol field. In non-thinking mode, preserve explicit final-answer markers and natural EOS, and label visible explanation tokens separately from absent internal-thinking tokens. Never salvage truncated or timed-out answers.

Choose development cohorts from the saved randomized training manifest before seeing their outcomes. Do not select only the two previously easy questions, drop slow questions, choose questions after decoding, or use official test answers during this stage. Run a capacity pilot at finite offered load and choose practical SLOs before the main search. If provisional SLOs change, disclose both the old pilot and the new adopted scope.

## Quality constraint and why 32 pairs are inadequate for a 2-point claim

Keep the original allowed accuracy drop of 0.02 absolute. A descriptive paired percentile bootstrap on a small cohort is not a proof of noninferiority; the old implementation's rejection of all degenerate bootstraps also makes identical outcomes impossible to certify even when enough independent questions exist.

A conservative preregistered alternative uses harmful paired discordance. For one sampled response per question, define harm as reference-correct and candidate-incorrect. Let `h` be the number of harmful questions among `n` independent paired questions. If `U(h,n,alpha)` is the one-sided Clopper–Pearson upper confidence bound on harm probability, then the population accuracy difference is at least `-U`, because candidate gains can only help. This deliberately ignores gains and is conservative. For repeated seeds, first aggregate within question and use the indicator that its paired mean difference is negative; all seeds remain clustered.

For zero harmful questions, `U = 1 - alpha**(1/n)`. With one-sided 95% coverage, zero harms in 32 pairs gives U approximately 0.0894, which cannot certify a 0.02 margin. Zero harms in 150 pairs gives U approximately 0.01977 and can pass. At 256 pairs, the exact upper bound can tolerate one harmful question within a 0.02 margin. The existing independent development-validation split has 256 distinct questions. Prefer a fixed validation cohort chosen in advance. Do not inspect 150 results and extend opportunistically without a preregistered sequential-testing correction.

The adopted continuation gate uses **exactly 150 distinct paired development-validation questions, one-sided 95% confidence and the original 0.02 margin**. There is no adaptive extension to 256. Freeze the cohort IDs and seeds before either validation arm. Save `study.quality_gate = {"method": "exact-paired-harmful-discordance-v1", "question_count": 150, "confidence": 0.95, "margin": 0.02}` in the adopted plan before validation. The implementation stores the exact gate under `independent_validation.paired_noninferiority` and the unchanged descriptive bootstrap under `independent_validation.paired_accuracy_ci`. The final validator recomputes the exact bound from the recorded harm count, checks the fixed cohort size and rejects mismatched or forged bounds before loading official test answers.

**Preregistered distinction between permission to measure and a statistical quality claim:** before any live search or independent-validation outcomes, the coordinator adopted `study.quality_gate.execution_criterion = "empirical-paired-accuracy"`. Execution of the held-out experiment requires the independent measured paired accuracy difference to be at least -0.02, an error rate at most 0.01, positive reference/candidate accuracy, the complete frozen 150-question cohort, and other measured feasibility checks. The original request requires actual development-supported accuracy constraints and paired confidence intervals; it does not require a 95% proof of noninferiority as a prerequisite to collecting held-out evidence. The earlier mandatory exact-bound gate was stricter than that request. The 0.02 margin is unchanged, and no outcomes were consulted to adopt this distinction.

The exact harmful-discordance bound remains the condition for the claim `statistical_noninferiority_established`. An empirical execution pass with a wider bound is labelled `empirical_pass_statistical_noninferiority_inconclusive`; it does not establish population-level preserved quality. Reports must retain this limitation alongside the paired bootstrap and exact interval. If empirical quality fails, no test is accessed and no replacement candidate or extended validation cohort is tried. Protocols omitting the explicit execution criterion retain the original strict exact-bound prerequisite.

After the eight-question GPU baseline and before search/independent validation, the coordinator specified a validation arrival rate of 0.6 requests/second to match the declared main load. Each arm uses the same 150 questions once, arrival `i / 0.6` seconds, no warmup/ramp, a 300-second measurement window, a 180-second per-request timeout and a 180-second maximum drain. The last question arrives at approximately 248.33 seconds. This replaces the provisional 4-requests/second validation trace before outcomes, avoiding an unplanned overload stress test as the main quality gate. Raw completion/error evidence and the full cohort remain retained.

The exact gate is a separate inference method, not a falsely labelled nondegenerate bootstrap. Save its method, one-sided confidence, loss count, question count, bounds, paired accuracies and hash-identified cohort/seeds. Continue reporting the usual paired bootstrap descriptively for the final effect. The exact bound guarantees coverage under its independent-question Bernoulli model; it does not remove benchmark contamination, device variability or sample-selection limitations. [R official exact binomial documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html), [SciPy exact binomial interface](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.stats.binomtest.html).

## Budget planning recommendation

Keep all startup, failed runs, warmup, ramp, drain, shutdown and idle reservation time in the original ledger. A provisional allocation that leaves room for completion is:

| Stage | Proposed maximum accelerator-seconds | Interpretation |
|---|---:|---|
| Already spent | 571.18 | Preserve exact ledger amount rather than charging this rounded display |
| New baseline/decoding/load pilot | 2,500 | Establish completion and useful quality before freezing serving workload |
| Live random search | 3,600 | Same shared starting design and actual time budget as other optimizers |
| Live GP search | 3,600 | No extra observations or table lookups |
| Live neural search | 3,600 | Charge failed runs/restarts and any optimizer time while GPU remains reserved |
| Independent quality validation | 5,000 | Conditional on observed output lengths; expand only by reducing another stage before it starts |
| Four held-out fixed-compute arms | 6,000 | Example 1,500 allocated seconds each; cohort and allowances frozen before test access |
| Uncommitted failure/extension room | 3,928.82 | Total remains below original 28,800 seconds |

These are planning allocations, not measured costs or a guarantee that 256 paired questions fit 5,000 seconds. Use development token lengths/throughput to select a feasible fixed cohort and arm budget before final freeze. A smaller test cohort such as 32 or 48 questions can be a valid fixed-compute experiment while retaining wide paired uncertainty. Do not call its P99 a robust tail estimate. Search seed replication is secondary to completing all three live optimizers with equal information and actual time.

## Equal compute means a reservation boundary

The existing `run_fixed_compute` uses a common post-readiness service deadline and reports unmatched startup/shutdown cost as a frontier. That is honest but does not fully satisfy the user's explicit equal-total-resource continuation request.

For the completed fixed-compute test, preregister one whole-reservation budget per arm measured from server spawn. Subtract observed startup and a fixed teardown allowance to determine the remaining service deadline. After stopping generation/server, retain the device's allocation in the serial experiment queue until the common reservation boundary; charge any padding as idle reserved-device time. This is consistent with the requested resource definition. Never clip an actual overrun to the target: retain it and qualify cost matching. Record startup, service, shutdown and idle padding separately. The final denominator includes every question in the frozen cohort, including undisbursed, timed-out and incomplete questions.

Similarly, identical fixed trial reservations or whole-optimizer reservation horizons can give equal actual tuning accelerator-time. An optimizer that finishes its allowed trials early can retain a charged idle allocation; this must not be represented as active useful compute. Plot best feasible observed goodput against both actual cumulative cost and trial count. A method can lose by spending its budget without improving the baseline. No method receives measurements collected for another method without those initial observations being identically charged/accounted for all methods.

## Freeze and claim discipline

Before official test access, freeze model/tokenizer revision, prompt/decoding mode, output allowances, answer parser mode, SLOs, selected static configurations, cohort IDs, per-question seeds, exact whole-reservation deadlines, arm order/randomization and analysis. Select official test indices using a predeclared seed before reading test content; retain all chosen questions. No test outcome may change later arms, parser rules or selected configurations.

When `final.test_cohort_seed` is defined, `freeze_protocol` shuffles the 1,319 official test IDs using that seed without opening test content; the first `final.question_count` IDs are used. The validator checks that their order matches the declared seed. The continuation uses seed 724. Protocols without this optional field preserve the original official ID ordering.

Use the default as a valid candidate in every method's starting design. If independent validation finds no improved feasible configuration, report that result. Do not manufacture a different tuned configuration merely to label an improvement. If the selected incumbent equals the default, disclose the identity and interpret its held-out repetitions as a control; a larger output allowance may still expose an accuracy-cost trade-off, but it does not establish a serving improvement.

No requirement in this document authorizes synthetic results to fill measured tables, omission of difficult requests, suppression of required reasoning, post-hoc margin changes, or payment for remote resources. The coordinator owns the serial accelerator queue and final budget decisions.
