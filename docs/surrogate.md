# Neural surrogate and fair search implementation

This is a supervised performance model. Its learned weights predict observations from complete serving trials; neither sampled answers nor measured latency is differentiated through vLLM. Candidate selection is discrete. The original deliverable contains measured-pilot diagnostic and separate synthetic demonstration checkpoints. The continuation trains new models only on its own measured development observations; earlier pilot rows and synthetic tables are not supplied to live policies.

## Inputs and split discipline

`surrogate.py` explicitly allows nine numeric features: scheduler batched-token limit, scheduler sequence limit, memory fraction, TP size, offered arrival rate, known input mean/p95 token length, requested output allowance, and burstiness. Precision and workload profile are one-hot features with an unseen-category bucket. Actual output lengths, test accuracy, latency, memory measurements, and row IDs never enter the features. Hardware is fixed within a study, so there is no learned cross-hardware claim.

A default control that omits scheduler flags must have its effective values resolved from the server before entering the surrogate. Missing values raise an error; the code does not invent defaults. Study code must retain the originally issued command separately from resolved feature values.

`grouped_split` connects repeated configuration/scenario pairs and any shared run or trace ID. A connected component cannot cross training/validation. Explicit validation partitions are honored and checked. Otherwise a seeded 80/20 group split is used. The feature encoder fits only training rows. For a zero-variance numeric feature, the scale is the larger of its absolute training mean and one; this avoids arbitrary count-unit amplification without consulting held-out values. Such a feature still has no learned extrapolation slope. Holdout or test records raise an error in fitting and search. Synthetic and measured records cannot be mixed. The intended external generalization test is an unseen load/length mix: call `predict` on it only after fitting. Internal early-stopping validation is not that final test.

## Model and losses

Three independently initialized MLPs share architecture, not parameters: input → 64 ReLU → dropout 0.05 → 32 ReLU → dropout 0.05. Separate linear heads produce two regression values and one feasibility logit. AdamW uses learning rate 0.001 and weight decay 0.0001; gradient norm is capped at 5. Default training stops after at most 200 epochs or 20 epochs without improved grouped validation loss. Selected states are restored. CPU execution does not reserve an inference accelerator on its own.

For trial i, the regression targets are `log1p(quality_goodput_i)` and `log1p(p99_e2e_seconds_i)`. A true zero goodput remains zero. A missing or censored P99 has mask zero; it is not imputed as a short latency. For the continuation, each log target is centered and divided by its own standard deviation computed over unmasked training entries only. This prevents goodput near 0.01 and latency near hundreds of seconds from producing very different residual scales. Validation targets use those frozen training statistics. A constant head uses scale max(abs(training mean), 1); an entirely missing head uses center zero/scale one and remains untrained. The loss is the mean of the two masked, per-head Huber losses on standardized log targets (delta 1), plus 0.25 times feasibility binary cross entropy. Failure labels must be retained by the caller. Training reports include available target counts. A latency head with no valid targets is untrained and cannot justify any claim.

Predictions first invert their head-specific training transformation and then use nonnegative `expm1`. The three member means and standard deviations describe model disagreement. They are **not calibrated confidence intervals**. Checkpoint format 2 contains architecture settings, train-only feature normalization/categories, target centers/scales, selected parameters, split keys, losses and target counts; loading uses PyTorch's restricted `weights_only=True` mode. Original format-1 checkpoints load with identity target transforms and retain their previous prediction semantics. Old saved policy files retain unscaled-target training on resume; new live policies explicitly save `neural_target_standardization=true`. The original pilot-diagnostic reproduction script explicitly requests the legacy transform, so the earlier experiment is not silently redefined.

Fewer than 20 independent connected groups triggers a small-data warning; repeated configuration/scenario pairs and shared run/trace identities can reduce that count. Six observed configuration/scenario groups is the default threshold before neural-guided selection begins, with random proposals before that point. That threshold is a software starting rule, not evidence that six groups are enough for useful learning. Actual prediction error and comparison with the GP must decide whether this model helps.

## Acquisition and constraints

Known backend/configuration constraints are checked before `SearchPolicy.choose`. For an untried candidate x, the neural acquisition is:

`A(x) = p_feasible(x) × [mean_goodput(x) + beta × std_goodput(x)]`, with `beta = 1`.

An optional declared P99 limit multiplies this score by a normal-CDF ranking factor from ensemble latency mean/spread. The default request-level SLO objective already includes latency compliance and does not need this extra population-level constraint. The factor, when used, remains heuristic. Predicted feasibility is never acceptance: selection must be followed by a real benchmark, actual development answer scoring, memory/error checks, and independent validation. No predicted accuracy head is used as proof of acceptable quality.

The conventional baseline is Gaussian-process expected improvement in `log1p` goodput, implemented with scikit-learn `GaussianProcessRegressor`: constant × Matérn(ν=2.5) + white-noise kernel, target normalization, and log-marginal-likelihood kernel fitting. Infeasible observed trials receive zero utility. A GP classifier estimates feasibility if both classes occur; otherwise a Jeffreys-smoothed empirical rate is used. EI is multiplied by that feasibility probability. Optional P99 constraint probability uses a separate GP on uncensored latency records. This is a conventional constrained GP baseline, not a reproduction of SCOOT.

Random search samples untried candidates uniformly. The default policy returns only the declared default once; it is a fixed control, not a searching optimizer. `initial_design` creates a shared diverse starting set from candidate features alone, beginning with the declared default if present. The orchestrator supplies the same initial observations and charges their cost to each simulated search. No policy can access an offline lookup table directly.

`measured_incumbent` compares mean measured goodput by configuration/scenario and requires every included repetition to be feasible. This conservative selection still needs an independent finalist run and a paired accuracy analysis. Returned predictions are never reported as speedups.

## Cost, resumption and interfaces

`SearchPolicy.choose(candidates, observed)` returns one candidate or `None`; `last_decision` records method, observations seen, acquisition, selected key, optimizer wall time and CPU time. The orchestrator owns the authoritative immutable observations and accelerator reservation ledger. It must enforce the same cumulative accelerator-seconds ceiling, account for failures/restarts, and log optimizer time as accelerator-time whenever devices remain reserved. Trial counts and actual cost must be plotted separately.

`save(path)` atomically stores the random-generator state and selected keys; neural state is also written to `path + '.pt'`. `load(path)` restores selections and RNG. Refit on the caller's unchanged observed ledger is deterministic with the same software and CPU settings. Pending selected trials are distinguishable from completed records; the orchestrator must reconcile them after a crash, rather than quietly rerunning a charged trial. Selected keys prevent duplicated proposals. Deliberate finalist repetitions are handled by validation orchestration, not by making the optimizer rediscover candidates.

Set `policy.artifact_directory` to retain each guided proposal's actual model separately. `closed_loop` automatically sets this to its `models/` subdirectory on new and resumed searches. Each `proposal-ORDINAL-HASH/` contains an actual `ensemble.pt` checkpoint, the exact already-observed input ledger, a report, and a SHA-256 integrity manifest. The report records parameter identity, the selected descriptor/prediction, full per-epoch training and validation losses, selected epochs, connected train/validation group identities, prediction audits, regression MAE/RMSE versus a training-only constant baseline, feasibility Brier/log loss, and descriptive reliability bins. Paths are recorded in `last_decision.model_artifact` and never enter policy settings or the frozen search contract. Files are published atomically, verified on deterministic retry, and never replaced by a later proposal; the most recent policy checkpoint remains available for compatibility. Artifact construction is included in optimizer CPU/wall timing.

These diagnostics assess the internal split used for early stopping. They are not an independent calibration set. With six observations, validation commonly has only one independent group, which cannot support reliable calibration, ranking accuracy or generalization claims. Reports flag fewer than five validation groups, a missing feasibility class, sparse uncensored P99 labels, and runs lacking at least 1,000 completed requests. Ensemble two-standard-deviation coverage is explicitly descriptive, not a calibrated interval. The prospective live adaptive outcomes and the separately frozen external test are the relevant evidence of whether learned predictions helped.

```python
from vllm_tuner.search import SearchPolicy, initial_design
from vllm_tuner.surrogate import NeuralSurrogate

starts = initial_design(valid_candidates, n=6, seed=7)
# Run and charge each start; only then append its result to observations.
policy = SearchPolicy('neural', seed=7)
policy.artifact_directory = 'results/models'
proposal = policy.choose(valid_candidates, observations)
# Run proposal; append immutable outcome and cost; checkpoint the policy.
policy.save('results/policy.json')
model = NeuralSurrogate(seed=7)
training_report = model.fit(development_observations)
heldout_predictions = model.predict(heldout_candidate_descriptors)
model.save('results/surrogate.pt')
```

## Prior work and attribution

Snoek et al. (2015), [DNGO](https://proceedings.mlr.press/v37/snoek15.html), established neural models for Bayesian optimization using adaptive basis-function regression. This project's small deep ensemble and optimistic acquisition are an implementation choice, not a reimplementation of DNGO or a claim to invent neural optimization.

Cheng et al. (2025), [SCOOT, arXiv v2](https://arxiv.org/abs/2408.04323v2), studies SLO-oriented inference-engine tuning with Bayesian optimization and constraint handling. Generic serving autotuning therefore is not this project's novelty. The proposed adaptation is answer-correctness-aware goodput, explicit fixed-fleet and tuning cost accounting, and a controlled small-model reasoning study. Original empirical findings are still required.

The GP implementation follows the documented [scikit-learn 1.6 Gaussian-process API](https://scikit-learn.org/1.6/modules/gaussian_process.html). Its predictive means/standard deviations and noise kernel support a conventional noisy black-box baseline. Kernel convergence warnings are suppressed during tiny-table fitting; fitted predictions and actual outcomes remain inspectable, and no warning suppression changes feasibility outcomes.

Sources were read on 2026-11-03. Tests exercise actual gradient updates, censor masks, split isolation, train-only preprocessing, safe checkpoint round-trips, observation visibility, fixed default behavior, candidate deduplication and RNG resumption. Automated tests and agent review do not substitute for course-required human peer review.

## Interrupted live searches

`experiment.closed_loop` freezes a search contract, records a proposal and advances its policy checkpoint before calling the expensive runner. If a proposal is pending on restart, it refuses to launch another trial. Reconcile the live reservation ledger first. If the live runner completed and wrote an immutable result but the search process stopped before recording it, use:

```python
from vllm_tuner.experiment import recover_pending
recover_pending('results/live/neural/search',
                'results/live/neural/observations.jsonl',
                'the-exact-completed-live-run-id')
```

The recovery function requires a matching completed measured development row, verifies its configuration, preserves the actual accelerator cost, appends it at most once, restores the incumbent summary, and clears that pending proposal. Calling it again for the same recovered run is idempotent. A failed or interrupted reservation without a completed live record first needs its actual resource cost reconciled and an auditable terminal live outcome; the helper will not invent one or silently repeat the run. An actual cost above the declared maximum remains charged and prevents subsequent search admission.

`trial_reservation_bound(study, protocol_override)` computes the common admission estimate in wall-seconds and allocated accelerator-seconds. It includes startup, all workload phases, graceful and forced shutdown waits, and bounded orchestration overhead. Use the same estimate in `run_trial` and the per-method search gate. It is conservative admission planning; actual elapsed resource cost remains authoritative.


## Live continuation comparison

The continuation keeps six initial configuration observations before neural acquisition. A compact ten-trial search therefore permits four genuinely adaptive live proposals per method. The six-group threshold remains a starting heuristic; it is not a statistical adequacy claim. Freeze the candidate space, workload, output allowance, input descriptors, decoding parameters, target-standardization version and acquisition settings before the first main comparison trial.

For strict equal actual tuning cost, independently benchmark the same ordered starting configurations for each optimizer and reserve equal-duration trial slots including initialization, warmup, measurement, drain and shutdown. CPU fitting can occur after releasing the device; if the reservation remains live during fitting or padding, that elapsed time must remain charged. Interleave methods by trial index to reduce wall-clock drift, without exposing another method's adaptive outcomes. Preserve the actual elapsed charge even when it differs slightly from a nominal slot.

A physically shared initial design is also possible, but its global physical charge and its common per-method information debit are different accounting quantities. Declare shared run IDs explicitly and report both. Never treat an adaptive result measured by one method as a free observation for another method.

`live_analysis.summarize_live_search(histories, common_initial_count=6)` independently checks the completed measured histories, common initial configurations, single environment/scenario, failures/costs and absence of undeclared adaptive reuse. Its incumbent comparison uses only observations completed before a common actual-cost checkpoint; it never looks forward to a cheaper-looking future result. One live search per method supplies no between-search-seed confidence interval. Finalist validation and official held-out scoring remain separate from these learning curves.

For matched per-index arrival schedules, pass `expected_trace_hashes=[digest(trace) for trace in planned_traces]`. The analysis requires a distinct, nonempty planned hash for every observed index and verifies every available actual trace hash against that common plan. Repeated within-method trace IDs are also rejected because they would connect the internal split. A terminal failed row without an actual hash remains in all cost and learning curves and is marked `unverified_failure`; its intended hash is reported separately and never relabeled as a measured trace. Successful rows without hashes and all actual/expected mismatches raise errors. Call `validate_planned_trace_hashes(hashes, required_count=trials_per_method)` before launching the campaign to catch duplicate or incomplete schedules without spending accelerator time. When no expected hashes are supplied, existing behavior is preserved and the report explicitly says trace matching was not audited.
