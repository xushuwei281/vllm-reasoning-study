# Presentation outline and oral-defence guide

## Nine-slide presentation, approximately ten minutes

1. **Question and success criterion — 1 minute.** Can a trained neural surrogate find a serving configuration that returns more correct GSM8K answers within request SLOs, under the same hardware and tuning allowance? Write `quality_goodput = correct AND all-SLO-compliant completions / measurement seconds`. State that a negative neural-versus-GP result is informative.
2. **Actual resources and scope — 1 minute.** Show the separate sanitized manifests: Apple M2 development diagnostics, then the measured main continuation on one NVIDIA A10G in a bounded Hugging Face job. Explain the distinct pinned Metal and CUDA runtimes and why their observations are never pooled for tuning. Show W8A8, TP>1, disaggregation and HCCL as NOT EVALUATED.
3. **Frozen reasoning protocol — 1 minute.** Qwen3-1.7B commit, GSM8K train/development versus official test, thinking sampling, prompt/parser hash, natural EOS, output allowance. Gold remains in the scorer. Explain what happens to malformed, truncated and timed-out answers.
4. **Measurement design — 1 minute.** Draw offered arrival → actual dispatch → first token → completion. Name client queue delay, TTFT, TPOT and end-to-end latency. Explain warmup/ramp/measure/drain, open-loop arrival scheduling, overload errors, and why completed-only P99 can hide poor service.
5. **Learned component — 1 minute.** Show configuration/workload descriptors → three small 64/32 MLPs → goodput, P99 and feasibility predictions. Explain measured supervised targets, masked censored latency, train-only normalization, and group isolation. The gradient changes surrogate weights, not vLLM startup flags.
6. **Search and comparison — 1 minute.** Same bounded candidates and the same six initial configurations, independently measured and charged for random, constrained GP and neural acquisition. State the acquisition formula and explain exploration from model disagreement. Plot best *measured feasible* goodput against both trials and actual tuning accelerator-seconds. Mark replay as replay.
7. **Evidence actually obtained — 1.5 minutes.** Use only generated figures and tables linked to raw run IDs. Separate real serving runs, synthetic orchestration tests and surrogate training. Report sample sizes, repetitions, missing telemetry, uncertainty and censoring. If full validation is unrun, state this on the slide rather than drawing a fictitious bar.
8. **Fixed-compute reasoning connection — 1 minute.** Default versus tuned × two frozen output allowances on the same held-out cohort. Plot conventional accuracy, deadline accuracy, actual tokens and allocated accelerator-seconds. If costs do not match, call this a measured frontier rather than an equal-cost win. If unrun, show protocol/status only.
9. **Contribution, limits and next experiment — 0.5 minutes.** The contribution is correctness-aware serving selection with auditable budgets and leakage controls, plus whatever original measured findings support. Cite SCOOT/DNGO/DistServe. End with the strongest supported finding and the exact next validation step, without a global-optimality claim.

For every result slide, put the measurement population, source (`measured` or `synthetic`), question count, repetition count and uncertainty method in the caption. Present limitations at the same time as the associated result.

## Questions every team member should be able to answer

**1. What is actually learned?** The MLP ensemble's weights map known serving and workload features to trial-level outcomes. The pretrained reasoning model is frozen. There is no gradient path through sampled answers, stopwatch measurements or `vllm serve` configuration changes.

**2. Why a neural network for such a small table?** It is a hypothesis, not an assumed advantage. Nonlinear interactions motivate trying it, but data scarcity may favor a GP or random search. The implementation warns about few independent points and reports held-out error. An actual comparison, not architecture complexity, determines value.

**3. What makes this different from existing autotuners?** SCOOT already tunes inference engines with Bayesian optimization; DNGO already uses neural models in optimization. Our proposed adaptation combines a correctness-aware objective, fixed-fleet/tuning accounting, a small reasoning model and controlled test isolation. Novel empirical findings require real measurements. Merely deploying these tools is not novelty.

**4. What does the acquisition optimize?** `p_feasible × (mean_goodput + beta × ensemble_std_goodput)`, with the declared beta. The standard deviation is model disagreement, not a calibrated confidence interval. An optional P99 ranking probability is heuristic. Every selection still needs real quality, memory and error validation.

**5. Why not backpropagate latency into the serving flags?** Flags are discrete/startup parameters and serving is an expensive noisy black box. We differentiate a regression loss to fit the surrogate; candidate evaluation and engine restarts are separate operations.

**6. What is the fair Bayesian baseline?** A documented constrained GP with a Matérn/noise kernel and expected improvement. It sees only observed permitted development rows, just as the neural policy does. It is not a claimed reproduction of SCOOT.

**7. What is the difference between a trial budget and a compute budget?** Trials differ in restart, warmup, timeout and reservation duration. Equal trial counts need not cost equal accelerator-time. We report both axes and enforce the declared actual resource ceiling, charging failures too.

**8. Does optimizer training count?** CPU optimizer time is always recorded. It consumes allocated accelerator-time whenever accelerators remain reserved during it. Releasing the server alone is not enough if a cloud reservation or exclusive deployment allocation continues.

**9. How can a one-device study discuss fixed fleet comparisons?** It can only claim results for that one device. TP-versus-replicas requires several real devices and the same total count on both sides. No multi-device conclusion is drawn from a single allocated Apple or NVIDIA GPU.

**10. Is the Metal `+cpu` core wheel a CPU benchmark?** Not by itself. The wheel supplies upstream vLLM and the external Metal plugin supplies the MLX GPU backend. Backend startup logs and live execution are the evidence, not the wheel name. Synthetic fake-server outputs are a separate software test.

**11. What does W8A8 mean here?** Eight-bit weights and eight-bit activations under an exact specified integer/float scheme, with documented scales, calibration and actual kernels. It does not imply an eight-bit KV cache or half the total memory. A weight-only MLX eight-bit checkpoint cannot be called W8A8.

**12. Why is HCCL absent?** HCCL is an Ascend collective library. Neither the local host nor the CUDA continuation has an Ascend stack. The documented buffer control cannot be repurposed as a generic vLLM or Metal switch, and KV-transfer connectors are a different mechanism.

**13. Why might prefill/decode separation lose?** Additional device pools, idle capacity, proxy processing, host buffers and KV transfer can outweigh reduced interference. A fair experiment must include the whole path and equal total allocation. We have no measured disaggregation finding.

**14. Why distinguish scheduler token budget from output allowance?** `max_num_batched_tokens` bounds work in a scheduler iteration. `max_tokens` bounds one response's permitted generation. Lowering the second changes the reasoning opportunity and cannot be silently used as a serving speedup.

**15. Why use Qwen3 thinking sampling rather than greedy?** Its model card recommends temperature 0.6, top-p 0.95 and top-k 20 for thinking. Sampling and seed protocol stay fixed across serving settings. The experimental Metal MPS path's greedy-only restriction is why the sampled study uses MLX.

**16. Are TTFT and TPOT engine measurements?** The primary values are client observed. TTFT starts at actual dispatch, excluding separately reported client queue delay. TPOT is `(completion − first_token)/(output_tokens − 1)` when there are at least two output tokens. Arrival-to-completion is additionally needed when the client queues.

**17. Can streaming chunks be treated as tokens?** No. A chunk may contain zero, one or many tokens. Save chunk times and tokenize with the pinned tokenizer or use trustworthy usage counts. Exact per-token latency within a multi-token chunk is unobservable; report chunk gaps or clearly labelled estimates.

**18. Why can P99 be misleading?** One hundred completions barely sample the upper tail. Timeouts can remove the slowest requests and make completed-only P99 look good. Report all offered requests, failure/censoring rates and independent repetitions; aim for at least 1,000 completed requests per configuration/profile for final tail claims.

**19. How do we avoid a closed-loop load generator?** Arrival timestamps are generated independently of completions. Record intended versus actual dispatch and any client rejection/queue. A semaphore must not quietly reduce offered load. Sustained tests have a steady arrival phase rather than one all-at-once burst.

**20. What prevents test leakage?** Official train data supplies calibration, profiling and selection; official test stays held out for final scoring. Grouped splits keep whole related runs/configuration-scenario repetitions together. Normalization learns from training rows only. No measured future output lengths or test outcomes enter deployment features.

**21. What does the accuracy denominator include?** The full preregistered offered/cohort population, with unanswered, invalid, truncated and timed-out responses handled as declared failures. Completion-only accuracy can be shown separately but cannot silently replace it. Paired question-level analysis clusters repeated seeds for one question.

**22. Can serving tuning improve model reasoning?** It may enable more correct completed answers or a more useful output allowance within a fixed budget. That is a system-level benefit. It does not change model weights or establish improved intrinsic reasoning. Actual tokens and matched costs must support the connection.

**23. Why is prediction not a result?** A prediction chooses the next measurement. Only a feasible measured candidate can become the reported incumbent, and short-run winners need independent validation to reduce selection bias. “Best measured configuration found within this space and budget” is the defensible phrase.

**24. How do you reproduce and audit a plot?** Start from its source run IDs and immutable raw responses/metrics, check environment/config hashes and reservation ledger, rerun the summary/plot script, then match the plotted population to the protocol. Synthetic fixtures cannot enter measured summaries.

**25. What if accuracy appears unchanged in a tiny sample?** Failure to detect a difference is not evidence of non-inferiority. Use the preregistered allowed drop with paired uncertainty. If the interval is too wide, statistical quality preservation remains unproven. The explicitly preregistered continuation permits held-out measurement after the empirical paired 2-point constraint passes, while separately reporting whether the exact 95% bound establishes noninferiority. It does not treat permission to measure as proof of preserved quality.

**26. When is tuning worth its cost?** Only when deployment savings and tuning costs have the same units. If tuning costs C accelerator-seconds and improves cost by d accelerator-seconds per correct accepted answer, break-even is C/d for positive d. If accuracy, SLO or workloads differ, or d is uncertain/nonpositive, do not quote a reliable break-even.

**27. Why do matched traces differ between trial indices?** The same actual schedule is used by all three methods at a given index, but fresh bounded jitter produces distinct schedules across indices. Whole repeated trace/configuration groups stay together. Reusing identical prompt content means the internal split estimates configuration generalization conditional on that workload, not unseen-content generalization.

**28. Why charge time after the server stops?** The cloud GPU remains allocated. Main trials have a 300-second nominal slot; idle padding makes the intended trial opportunity comparable. The original ledger also charges CPU optimization, setup and between-trial gaps. Final arms have a 360-second whole-reservation budget: startup reduces serving time, and actual overshoot is retained rather than rounded away.

**29. Does a fixed 360-second arm represent steady-state serving?** No. All 128 questions are offered at once after startup, so the result measures startup-inclusive deadline capacity on a finite held-out cohort. The separate search uses a 90-second open-loop arrival window. Fixed final arm order and persistent compilation caches are limitations.

**30. Where is evidence that the network was trained before its decision?** Each guided proposal preserves the permitted observations, train/validation identities, per-epoch losses, weights, selected prediction and checksum manifest before the selected run begins. Its proposed run's future outcome is absent. There are only 6–9 observations, so trained weights alone do not establish useful generalization.

**31. How is paid usage bounded?** One A10G job has a four-hour service-side timeout. The verified rate plus private-port fee bounds its cost below the authorized US$10, including the retained failed launch and minute-rounding margin. The original 28,800-second ledger is extended, never reset. Terminal allocation evidence is reconciled against every charged trial and overhead interval. This provider omitted its finish timestamp, so the delivery uses the explicitly labelled conservative charge through confirmed cancellation.

## Code walkthrough for a rehearsal

- `config.py` / `environment.py`: show exact release/revision pinning, known constraints and the distinction between installed/documented/live-tested capability.
- `lifecycle.py` / `accounting.py`: show start-to-stop reserved time, failure charging, and resumable budget records.
- `benchmark.py` / `evaluation.py`: trace one offered request through timestamps, stream chunks, terminal status, parser and quality-adjusted goodput.
- `surrogate.py`: show the allowed feature list, grouped split, loss masks, optimizer step and checkpointed weights.
- `search.py` / `replay.py`: show the acquisition, observation boundary, equal initial information, checkpoints and measured incumbent.
- `experiment.py`: show one real run's argv, raw outputs and ledger; then demonstrate how the same source data produces the report's summary.

A team member should rehearse explaining one failed trial and one incorrect answer. If those examples do not exist in saved real records, say so rather than inventing them.

## Completed terminal accounting note

The job is confirmed canceled, but Hugging Face omitted its finish timestamp. The delivered ledger therefore reconciles prior charges plus the conservative creation-to-confirmed-terminal-observation upper bound, not an invented exact duration. The charged total is 3.8879 accelerator-hours; estimated HF charges are at most US$5.6625 under minute rounding. The exact provider finish and invoice amount remain unknown.
