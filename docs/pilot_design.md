# Metal pilot diagnosis and bounded follow-up design

This document records the evaluation agent's staged review and the final observed pilot outcome. The exact saved configurations, request traces, question IDs, decoding seeds, and hashes are authoritative. The proposed nine-point sweep below was **not executed**. No official test answers were fetched or scored.

## Final observed stage outcome

Three development-only runs completed orchestration. They reserved a total of **571.1714282079483 accelerator-seconds** (about 0.1587 hours):

| Saved run | Measurement requests | Completed including drain | Correct / full cohort | Quality goodput | Accelerator-seconds |
|---|---:|---:|---:|---:|---:|
| `pilot-metal-default-001` | 4 | 0 | 0/4 | 0.000/s | 165.920426542 |
| `screen-metal-00` | 2 | 2 | 0/2 | 0.000/s | 145.663405208 |
| `allowance-metal-2048-00` | 2 | 1 | 1/2 | 0.010/s | 259.587596458 |

The sequence-limit-one diagnostic at 1,024 output tokens ended both responses with `finish_reason=length`; those are invalid reasoning completions even when latency conditions pass. The subsequent matched development intervention at 2,048 tokens produced one correct final answer at natural EOS after 1,478 server-counted output tokens; the second measurement request timed out. Its 50% error rate violates the declared 1% error limit. It is therefore **not a feasible finalist**. Different actual reservation costs make the allowance contrast a descriptive accuracy-cost frontier, not an equal-compute improvement.

The nine-point scheduler sweep, live optimizer comparison, held-out workload validation, final test scoring, and four-arm fixed-compute experiment remain **NOT EVALUATED**. The broader search stopped at the development completion/quality gate; it did not consume the available eight-hour ceiling merely to produce more rows. The two allowance runs share a matched trace and must remain in the same surrogate split group. Their two-question paired accuracy contrast is descriptive and cannot establish noninferiority. The initial pilot below is retained as the evidence that motivated the bounded follow-up.

## Evidence from the first real serving pilot

The first measured run, `pilot-metal-default-001`, used the Qwen3-1.7B reference checkpoint through the installed vLLM Metal backend on one Apple M2 GPU. It offered five requests across all phases, of which four arrived in the 40-second measurement cohort. All four measurement requests reached their 90-second timeout. None completed a valid final answer. Consequently, measured quality-adjusted goodput and full-cohort completed-answer accuracy were both zero for this pilot; completed-request latency percentiles were unavailable and the all-offered P99 was censored. This is a failure finding for that exact short pilot, not an estimate of the model's unconstrained mathematical accuracy or a universal backend limit.

The recorded reservation cost was **165.92042654199759 allocated accelerator-seconds**, including **28.89156770799309 seconds of startup**. Total collection wall time was 136.43671470799018 seconds. This is a single shared-memory GPU reservation, not a dedicated 16-GB GPU plus a separate 16-GB host. Device-memory allocation was not measured.

Sources are the immutable files in `results/measured/runs/pilot-metal-default-001/`: `observation.json`, `collection.json`, `scored.jsonl`, `requests.jsonl`, `server.log`, and `provenance.json`. The observation row's `status=ok` means orchestration completed and produced a valid failed-service measurement; its `metrics.feasible=false` and timeout/error counts describe the workload outcome.

The stream traces show generated text before timeout. Re-encoded partial reasoning spans approximately 695–766 tokens for the four measurement requests; these are text-based estimates, not authoritative server token usage. One partial stream crossed the closing thinking marker but timed out before a complete scored final response. That distinction matters: the first experiment demonstrated streaming execution and missed deadlines, not a failure to launch the model. Its total generated token usage is unknown because interrupted streams did not supply final usage.

## What the next pilot must resolve

At the first review, the immediate uncertainty was whether the frozen 1,024-token reasoning allowance permitted valid answers at lower contention and a longer request deadline. The subsequent results above resolve it only for the two tested questions: both truncate at that allowance. A scheduler search has little information if every configuration produces zero valid answers. Increasing the deadline from 90 to 180 seconds and lowering arrival rate from 0.1 to 0.02 requests/s are development-protocol changes, so the next results cannot be called a speedup over the first pilot. A reference run under the new protocol is required.

The proposed bounded nine-point space, retained here as an unexecuted plan, was:

| Variable | Values |
|---|---|
| `max_num_batched_tokens` | 256, 512, 1024 |
| `max_num_seqs` | 1, 2, 4 |
| `gpu_memory_utilization` | fixed at 0.5 |
| Precision, placement | reference BF16; TP=1; colocated |
| Reasoning output allowance | fixed at 1024 tokens |
| Request timeout | fixed at 180 seconds |
| Offered load | two scheduled measurement arrivals per 100 seconds, nominal 0.02/s |

Known backend validation and exact CLI help checks remain prerequisites. The server's maximum batched-token setting is not an answer-length cap. A sequence limit of one still accepts queued requests through the server and is distinct from limiting the load generator to one in-flight request.

This nine-point screen is sensible only as a small development feasibility diagnostic. Two answers per configuration cannot establish preserved quality, characterize P99, support a persuasive neural-vs-Bayesian-vs-random comparison, or establish generalization. One short trace cannot establish sustained-load stability. At most it can expose severe invalidity, timeout behavior, obvious queue effects, and a candidate worth measuring again.

## Recommended staged execution and stop rules

Do not launch all nine configurations or consume the eight-hour ceiling merely because it exists. Save the follow-up protocol before execution, then use a staged pilot:

1. **Completion diagnostic:** run the low-load reference once with the new deadline and fixed allowance. If necessary, run one capacity-limited diagnostic at batched tokens 512, sequences 1, memory 0.5. Use the same predeclared development questions and decoding seeds in both conditions. Do not select easier questions after seeing their answers.
2. **Early stopping:** if neither diagnostic yields a valid final answer, stop the serving search and record an insufficient-feasibility result. Inspect development truncation and timeout reasons. A later allowance/protocol change would be a new preregistered development stage with a new reference, not a continuation under unchanged conditions. No test fetch or final protocol freeze is warranted.
3. **Small scheduler screen:** if completion is demonstrated, measure at most three representative settings first, including the matched reference where applicable. Expand to the full nine points only if the data contain meaningful variation in completed correct SLO-compliant work and enough remaining budget for independent confirmation.
4. **Independent development confirmation:** re-run the best measured candidate and matched reference on another declared training-only cohort and independent timing trace. Failure to reproduce counts as a negative/inconclusive result. Keep the incumbent described as the best measured configuration within this tiny space, never globally optimal.
5. **Final gate:** only a supported development-quality result and an executable final budget justify freezing final configurations and accessing test data. Zero-versus-zero accuracy is not evidence that reasoning quality was preserved. Do not waive the paired quality gate to create a finalist.

The maximum reservation for each pilot must include its startup timeout, controlled phases, 100-second measurement, up to 180 seconds of drain, shutdown, and accounting margin. Admission is based on this bound, while the ledger charges actual reserved wall time. A nine-configuration sweep can therefore consume substantially more than nine times 100 seconds. The root runner must enforce a stage cap before admission; the total eight-hour ceiling is not an instruction to use the whole allowance.

## Trace and question controls

At two requests per condition, changing question cohorts across configurations confounds configuration effects with question difficulty. Prefer identical question IDs and sampling seeds across configurations within a stage. Draw independent deterministic arrival jitter for each measured configuration/trace, using a preregistered seed schedule matched across optimizer trial indices. This supplies distinct trace identities for honest grouped surrogate validation without silently swapping easier tasks into a fast configuration. Save every intended timestamp and the jitter distribution, not just a nominal rate.

A later independent cohort should be a separate confirmation stage applied to all compared configurations. All repetitions of the same configuration-workload scenario still remain together when splitting surrogate rows. If shared traces connect every row into one group, there is insufficient independent information for a train/validation split; report that limitation rather than randomly splitting requests or deleting trace identities.

The proposed two-arrival/100-second trace has strong measurement-boundary effects. A request arriving near the midpoint has only about 50 seconds to contribute to window goodput, although it may legitimately finish during the 180-second drain. Report primary window goodput exactly as defined alongside full-cohort accuracy, drain completion, and arrival-to-completion latency. Do not reinterpret correct drain completions as in-window goodput after seeing them. If all correct completions occur during drain, that is a reason to redesign a subsequent longer measurement window on development data, not to claim the original objective improved.

No arrival schedule should wait for previous completion. A deterministic periodic trace with declared jitter is a trace-replay diagnostic, not a Poisson sustained-load claim. At only two measurement arrivals, terms such as near-saturation and stable throughput are unsupported unless a separate longer run supplies evidence.

## Limits of learning and comparison at this scale

Nine configuration-workload rows, each based on two offered questions, are much too sparse for a reliable 64/32-unit neural performance model. The implemented ensemble can still train as a software demonstration, but its predictive uncertainty is not calibrated deployment risk. If all measured quality goodputs are zero, the optimizer has no observed positive objective signal; do not create positive labels from partial correct-looking thoughts or predictions.

A shared offline-table replay may compare random, GP, and neural selection mechanics if all methods receive identical initial information and each queried row is charged its recorded cost. Label replay explicitly, report the tiny independent sample count, and avoid a measured live-search speedup claim. A replay finalist still needs independent live confirmation. Synthetic data can validate the optimizer implementation but cannot supply free measured training examples or evidence that neural search outperforms the baselines on this hardware.

P99 claims require many more completed requests and independent sustained windows. A paired 2-percentage-point quality margin cannot be established from two questions. The two planned output allowances must be development-selected and frozen before test access; testing the fixed-compute reasoning comparison is premature while valid reference completions remain unestablished.

## Current interpretation

The observed evidence supports bounded feasibility findings on this M2 serving path: the initial configuration/protocol times out, the lower-load 1,024-token diagnostic truncates reasoning, and a 2,048-token allowance permits one correct completed answer while another request still times out. The follow-up ended with **no established feasible configuration and insufficient data for comparative optimization**. No optimizer advantage, quality preservation, memory saving, tail-latency improvement, or stronger intrinsic reasoning follows from these three runs.


## Final-experiment gate schema

`fixed_compute.validate_frozen_protocol` is shared by protocol freeze, test download, and final execution. A bare `quality_gate_pass=true` flag is not enough. Each selected record must be a feasible measured development observation and include separate independent validation evidence, for example:

```json
{
  "independent_validation": {
    "source": "measured",
    "partition": "development_validation",
    "quality_gate_pass": true,
    "run_ids": ["a-real-independent-validation-run-id"],
    "question_count": 128,
    "question_ids_hash": "<sha256 of the actual shared question IDs>",
    "sampling_seeds_hash": "<sha256 of the actual shared sampling seeds>",
    "metrics": {"feasible": true, "accuracy": "<actual measured value>", "error_rate": "<actual measured value>"},
    "paired_accuracy_ci": "<actual paired_accuracy_ci output, required for the tuned candidate>"
  }
}
```

The angle-bracket values above are schema placeholders, never measured evidence. Keep the actual independent raw requests and run records. Both arms require distinct validation run IDs, matching cohort/seed hashes, at least 30 question clusters, a positive measured reference accuracy, and errors inside the frozen limit. The candidate also requires a usable, nondegenerate paired 95% interval whose lower bound is no worse than the frozen accuracy-drop margin. The 30-cluster warning threshold is not a statistical power guarantee. The reference is independently remeasured; it does not need an impossible nondegenerate comparison with itself.

The validator also checks the freeze digest, official data revision, parser version, lack of prior test results, unique official test IDs, frozen positive allowances/deadline/allocation, and sampling seeds. Final execution must use the development environment hash, apply the frozen sample seed to actual generation, and preserve identical questions across all four arms. A per-seed plan saves prompt hashes without gold. Output summaries and paired comparisons are named `fixed-compute-summary-seedSEED.json` and `fixed-compute-comparisons-seedSEED.json`. Invalid or startup-failed arms retain the entire unanswered cohort and their actual cost, including a legitimate zero when no accelerator reservation began. No such final experiment has run in this project.
