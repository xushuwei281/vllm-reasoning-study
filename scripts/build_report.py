"""Regenerate the research report and diagnostic figures from immutable evidence."""
from pathlib import Path
import json
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from vllm_tuner.data import read_jsonl,write_json
from vllm_tuner.accounting import BudgetLedger
from vllm_tuner.plots import measured_figures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=read_jsonl(root/'results/measured/observations.jsonl')
by={r['run_id']:r for r in rows}
labels=['Default, 1,024','Seq 1, 1,024','Seq 1, 2,048']
ids=['pilot-metal-default-001','screen-metal-00','allowance-metal-2048-00']
cost=sum(r['cost']['accelerator_seconds'] for r in rows)
test_text=(root/'results/software-tests.txt').read_text()
import re
tests_passed=int(re.search(r'(\d+) passed',test_text).group(1))
paired=json.loads((root/'results/measured/diagnostic-paired-accuracy.json').read_text())
synth=json.loads((root/'results/synthetic/demo/model/training.json').read_text())
measured_figures(root/'results/measured')
fig,axs=plt.subplots(1,2,figsize=(10,4.5))
for j,key in enumerate(ids):
 r=by[key];m=r['metrics'];n=m['offered'];correct=round(m['accuracy']*n);trunc=m['truncated_count'];timeouts=m['status_counts'].get('timeout',0)
 bottom=0
 for name,value,color in [('Correct final',correct,'#238b76'),('Truncated',trunc,'#e8af39'),('Timeout',timeouts,'#c44b52')]:
  axs[0].bar(j,value,bottom=bottom,color=color,label=name if j==0 else None);bottom+=value
 axs[1].bar(j-.17,m['latency_goodput'],.32,color='#6699cc',label='Latency-only' if j==0 else None)
 axs[1].bar(j+.17,m['quality_goodput'],.32,color='#238b76',label='Correct + all SLOs' if j==0 else None)
for ax in axs:ax.set_xticks(range(3),labels,rotation=15,ha='right');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
axs[0].set(ylabel='Measurement-cohort requests',title='Observed outcomes (small development pilots)')
axs[1].set(ylabel='Completed answers / measurement second',title='Why correctness changes goodput')
fig.text(.5,.01,'MEASURED. Loads and deadlines differ across the first and later pilots; no optimizer comparison.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.08,1,1));fig.savefig(root/'results/measured/pilot-outcomes.png',dpi=180);plt.close(fig)

summarylines=[]
for label,key in zip(labels,ids):
 r=by[key];m=r['metrics']
 summarylines.append(f"| {label} | {m['offered']} | {m['completed']} | {round(m['accuracy']*m['offered'])}/{m['offered']} | {m['quality_goodput']:.3f} | {r['cost']['accelerator_seconds']:.2f} |")
table='\n'.join(summarylines)
report=f'''# Learning to Tune vLLM for Compute-Budgeted Small-Model Reasoning

**Implementation and bounded local feasibility study - 3 November 2026**

## Abstract

We implement a quality-constrained serving optimization pipeline using vLLM, a compact PyTorch performance-surrogate ensemble, random search, and Gaussian-process Bayesian optimization. The intended objective is correct, completed reasoning answers satisfying request-level latency SLOs per measurement second, under fixed allocated accelerator resources and a fixed tuning budget. We discover and verify a local Apple Metal backend rather than assuming that an Apple host cannot run vLLM. We execute three small development experiments with Qwen3-1.7B and GSM8K training questions. The initial load produces four measurement-request timeouts. A lower-load, single-sequence diagnostic completes two requests but truncates both at 1,024 tokens. Raising the allowance to 2,048 produces one correct natural-EOS answer and one timeout. Total reserved accelerator-time is {cost:.2f} seconds ({cost/3600:.4f} hours).

These are feasibility results, not a completed optimizer superiority study. A real three-member surrogate is trained on the measured pilot records only as a diagnostic, and the complete search loop is exercised separately on explicitly synthetic fixtures. The profiling sample is too small, contains no established feasible configuration, and has only two independent trace groups. No measured search winner, held-out GSM8K result, P99 service guarantee, quantization gain or improved intrinsic reasoning is claimed. The deliverable includes executable code, tests, pinned environments, raw records, figures, report-generation scripts and a staged continuation protocol.

## 1. Motivation, question and contribution

Serving efficiency matters for small reasoning models because waiting time, queueing and output truncation can prevent correct answers from arriving within a resource envelope. Token throughput alone can reward a server that produces incomplete or incorrect answers. We therefore ask: can a small neural surrogate discover a valid vLLM configuration with higher quality-adjusted goodput than default settings, random search and conventional Bayesian optimization at equal tuning accelerator-time?

The contribution implemented here is a workload-specific adaptation: strict final-answer evaluation inside request-level SLO accounting; explicit measurement/drain and timeout rules; whole-run and trace-isolated surrogate fitting; shared cost accounting across optimizers; and a gated four-arm output-allowance experiment. This is not a new inference engine, a model-training method, a question router or a claim of globally optimal serving. Original local findings concern the mismatch between apparent completion/latency efficiency and usable reasoning answers. The primary comparative research question remains unanswered by the available pilot evidence.

SCOOT already establishes serving-engine autotuning with Bayesian optimization and feasibility constraints. DNGO establishes neural surrogates for expensive black-box optimization; our small ensemble is not a reproduction of DNGO's Bayesian linear output layer. DistServe establishes the motivation for separating prefill and decode under SLOs. Those precedents limit algorithmic novelty; none proves a benefit on this small model or host. [SCOOT](https://arxiv.org/abs/2408.04323v2), [DNGO](https://proceedings.mlr.press/v37/snoek15.html), [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).

## 2. Model, data and environment

The reference checkpoint is Qwen/Qwen3-1.7B, Apache-2.0, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The client tokenizer uses the same revision. The prompt requests reasoning and a boxed numeric final answer. Thinking mode is enabled, temperature 0.6, top-p 0.95, top-k 20 and min-p 0, following the model card. Natural EOS remains enabled; the main comparison must freeze decoding and output allowance. The local diagnostic explicitly changes the allowance as a separate development intervention. [Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B).

GSM8K comes from official repository commit `3101c7d5072418e28b9008a6636bde82a006892c`. Stable line-index IDs and a deterministic seed partition 128 training questions for calibration, 512 for profiling and 256 for independent development validation. Gold answers stay in the evaluator; they never enter generation prompts or repair steps. The official 1,319-question test split is not fetched or scored in this study. Test download is gated on a frozen development selection. This project-level separation does not rule out contamination during pretrained-model training. [GSM8K repository](https://github.com/openai/grade-school-math).

The host is an Apple M2 MacBook Air with one 8-core Apple GPU, 8 CPU cores and 16 GiB unified memory, running macOS 27.0.1 on arm64. Host and GPU memory are shared, not additive. The isolated serving stack is Python 3.12.12, vLLM 0.32.0+cpu, vLLM Metal 0.32.0, MLX 0.32.2 and torch 2.13.0. The upstream CPU wheel hosts an external Metal/MLX GPU execution plugin. Startup logs confirm the MLX GPU device and single-rank Gloo initialization. No NVIDIA device, Ascend NPU, NCCL/HCCL topology or cloud fleet is assumed. The analysis/client environment uses Python 3.11.4 and torch 2.7.1 separately.

Installed `vllm --version`, `vllm serve --help=all` and `vllm bench serve --help` were captured. Model execution, streaming completions and `/metrics` were observed. The custom harness is justified by correctness scoring, exact boundary handling and immutable request traces; stock SLO goodput does not incorporate answer correctness. [Metal release](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0), [versioned serve CLI](https://docs.vllm.ai/en/v0.32.0/cli/serve/), [benchmark CLI](https://docs.vllm.ai/en/v0.32.0/cli/bench/serve/).

| Feature | Verified local treatment |
|---|---|
| Reference precision | BF16 executed; FP16 capability documented but not compared |
| Scheduler controls | Chunked prefill on; batch and sequence limits accepted by installed CLI |
| Prefix caching / KV dtype | Prefix caching explicitly off; KV cache dtype auto/reference dtype |
| Memory control | Fraction 0.5 in live pilots; effective KV capacity can change under OS pressure |
| W8A8 | NOT EVALUATED: no verified small-checkpoint activation/kernel path on Metal |
| TP versus replicas | NOT EVALUATED: one device; Metal rejects TP greater than one |
| Prefill/decode disaggregation | NOT EVALUATED: no compatible verified split deployment/device pool |
| HCCL | NOT EVALUATED: no Ascend/CANN/torch_npu/HCCL installation |
| Memory time series | Device peak bytes unavailable; raw cache ratios and startup allocation logs retained |

The original pilot log records a pressure-related KV capacity reduction from 2.35 GB to 0.93 GB. Those are backend planning figures, not measured peak GPU residency or a saving caused by tuning. The backend's pressure probe was not disabled. Details and primary sources are in `docs/capabilities.md` and `docs/systems_sources.md`.

## 3. Budgets, constraints and objective

Deployment allocation is one Apple GPU for each live reservation, including startup, compilation, idle time, load generation, drain and shutdown. The provisional study ceiling is 28,800 allocated accelerator-seconds (eight hours), with 25% reserved for validation/failures. A single shared ledger admits only one active reservation. Costs are actual reserved devices multiplied by wall time, never summed request latencies or active FLOPs. CPU-only fitting occurs after server shutdown and is reported separately. Installation/download time is not accelerator time because no model server reserves the device then.

The pilots consumed {cost:.2f} allocated accelerator-seconds. The ceiling was not exhausted; the broader sweep stopped at the development quality/completion gate. At 0.1 offered requests/s, even 1,000 arrivals take 10,000 seconds before repeats. The aspirational repeated final-tail design therefore cannot be presumed to fit the two-hour validation reserve. It must be revised using pilot capacity and precision requirements before final freeze.

Provisional request SLOs are TTFT <= 5 s, TPOT <= 0.10 s and dispatch-to-completion <= 180 s. The initial pilot has a stricter 90 s timeout; subsequent diagnostics use 180 s. Allowed accuracy drop is 0.02 absolute relative to an appropriate reference, with error rate <= 0.01. These are provisional research choices, not validated user-service requirements. No quantized quality gate was evaluated.

The primary objective is:

`QAG = count(correct, natural-EOS, all-request-SLO-pass completions inside the measurement window from its arrival cohort) / window seconds`.

Dividing by the allocated device count gives quality-adjusted goodput per accelerator, equivalently compliant correct answers per allocated accelerator-second for that measurement window. Whole-trial accelerator cost, including startup/drain, is reported separately. Memory, backend support, quality loss and error rate remain constraints, not weighted terms hiding trade-offs.

## 4. Instrumentation and experimental protocol

Arrivals are scheduled independently of request completion. The initial pilot uses a seeded finite-rate Poisson trace with 5 s warmup, 5 s ramp and a 40 s steady window at target 0.1 requests/s. The later diagnostic pair uses the same two training questions and decoding seeds, two 64-token warmup/ramp requests excluded from quality scoring, and two timestamped low-load measurement arrivals in a 100 s window. Its 1,024- and 2,048-token allowances are explicitly distinct experimental conditions. The paired trace is identical, so both runs remain one group in surrogate splitting. The single-sequence configuration fixes batched-token limit 512, memory fraction 0.5 and TP=1.

Every offered request has intended arrival, actual dispatch, first text-bearing streaming chunk, completion/observation end, status, finish reason, text and server usage when available. A concurrency cap rejects explicitly rather than silently closing the load loop. A total-stream timeout prevents indefinitely arriving chunks from extending request deadlines. Raw offered and completed records are append-only. Supplied traces are checked for ordering, phase boundaries and a finite offered window before admission.

TTFT is observed from dispatch to the first nonempty generated text chunk. Transport buffering or multi-token chunks can make it a delayed observation of the first token. TPOT is `(completion - first_chunk)/(output_tokens - 1)` for at least two server-reported output tokens. Chunk gaps are saved; true within-chunk token timing is unavailable. Arrival-to-completion and client dispatch delay are separate metrics. Throughput is labelled completed requests/s, output tokens/s or input tokens/s. Partial timed-out output does not become a fabricated authoritative token count.

The arrival cohort is the half-open steady window. Primary goodput requires completion before its end; a later drain completion contributes to full-cohort accuracy but not the window numerator. Timeouts, errors, rejections, truncations and unanswered questions stay in their denominators. All-offered P99 puts failed requests in the censored tail; completed-only quantiles are separate diagnostics. A P99 calculated from two completions is only an observed order statistic, not evidence for a deployment tail.

The answer parser accepts explicit final numeric markers/boxed answers, normalizes exact rational values, and rejects malformed, unfinished-thinking, timed-out or length-truncated outputs. A correct-looking number inside an unfinished thought does not count as a completed answer. Reasoning/final token counts re-encoded from text are labelled approximate partitions of text, distinct from total server usage.

## 5. Surrogate and search implementation

Each dataset row is configuration x workload x repetition, with environment hash, predeployment descriptors, measured metrics, failure status and actual costs. Inputs include batch/sequence/memory/TP settings, precision, profile, offered rate, prompt-length descriptors, output allowance and burstiness. Future output lengths and held-out outcomes are excluded. All repeated measurements of a configuration/scenario and linked run/trace identities stay together. Scaling and categorical levels are fitted on training records only.

Three independently initialized PyTorch MLPs have hidden widths 64 and 32, ReLU and dropout. AdamW trains masked Huber regression on log1p goodput/P99 and binary cross-entropy for operational feasibility. Zero goodput is valid; censored P99 is masked rather than replaced with fast latency. Grouped validation controls early stopping. Checkpoints include learned weights and preprocessing. Missing/constant features and unseen categories have explicit handling. The target server is an expensive noisy black box; no gradient passes through serving latency or sampled answers.

The neural acquisition ranks valid, untested configurations using:

`A(x) = p(feasible | x) * (mean_QAG(x) + beta * sd_QAG(x)) * optional_P(P99 <= limit | x)`.

The ensemble standard deviation is an exploration heuristic, not a calibrated confidence interval. Actual development answer quality and live measurement are required for acceptance. With fewer than six independent configuration/scenario observations, the search policy records a random warm start rather than pretending to perform informed neural selection.

Baselines are the documented default, random sampling and constrained GP expected improvement with a Matérn-5/2 kernel and observation noise. Each search sees only its revealed, charged observations. Shared starting designs, per-method time caps, common global accounting and number-of-trials curves separate information budget from accelerator cost. The offline replay holds the same initial information and reveals one saved repetition at a time; full-table diagnostic training is separate and is never given to a search policy for free.

The live loop restarts startup-only configurations, persists pending proposal/RNG state before launch, checks immutable input fingerprints, charges failures, and stops if an actual cost overruns its declared admission bound. Explicit recovery reconciles completed live rows; it never invents outcomes or silently repeats interrupted expensive trials. CPU optimizer cost is stored separately when the GPU is released.

## 6. Measured development results

| Condition | Offered in window | Completed incl. drain | Correct / full cohort | Quality goodput, answers/s | Whole-trial accelerator-s |
|---|---:|---:|---:|---:|---:|
{table}

![Measured pilot outcomes](results/measured/pilot-outcomes.png)

The initial default uses resolved API scheduler defaults (2,048 batched tokens and 256 sequences, source-verified for this installed Metal platform) with controlled memory/context/precision settings. It launches successfully and emits text, but all four measurement requests time out. Their all-offered P99 is censored and total generated tokens are unknown because terminal server usage is missing.

At lower load and one sequence slot, both 1,024-token responses complete with `finish_reason=length`. One completes inside the window and satisfies the latency-only checks, giving 0.010 latency goodput, yet both are invalid reasoning completions and quality goodput remains zero. The two completed E2E latencies give an observed P99 value of 66.80 s; n=2 makes it unsuitable as a tail-latency claim.

At 2,048 tokens, the first question produces a correct boxed answer, 7, after 1,478 output tokens and natural EOS. The other request times out at 180 s even though its partial text contains the expected numeric answer. It is correctly scored as failure. Full-cohort accuracy is 1/2 and quality goodput is 0.010 answers/s. Error rate is 1/2, far above the 0.01 constraint, so this is not a feasible finalist. All-offered P99 remains censored.

The paired two-question allowance comparison has descriptive accuracy difference +0.50 and a nominal paired bootstrap 95% interval [0.00, 1.00]; the evaluator explicitly marks its quality gate unusable at two question clusters. It cannot establish noninferiority or an accuracy gain beyond these questions. The 2,048-token run also uses more reserved accelerator-time than the 1,024-token run. This is a development accuracy-cost frontier point, not a same-compute improvement or the requested held-out four-arm experiment.

The three runs contain eight measurement-cohort request instances, with repeated questions and different protocols. They must not be pooled as eight independent benchmark tasks or deployments. There are no independent repeats sufficient for a latency confidence interval. No confidence interval is fabricated for censored P99.

![Measured accuracy versus actual reservation cost](results/measured/accuracy-cost.png)

## 7. Neural diagnostics and software validation

The actual measured pilot table has three rows but only two independent trace groups: the two allowance conditions share the same trace. Default scheduler values are resolved in a derived copy with source evidence; raw records are never changed. A diagnostic fit keeps the allowance pair together for training and the default pilot as one internal validation group. This has only two training goodput labels, one uncensored P99 label, and no positive feasibility labels. The output checkpoint is evidence that the neural path executes on real measurements, not a useful validated performance model. Its internal validation goodput MAE is 0.03791, compared with 0.00500 for the training-mean constant; with just one zero-goodput validation record, neither is a generalization estimate. P99 has no validation target, and no live acquisition decision is justified by this fit. See `results/measured/pilot-model/diagnostic-summary.json` for exact fit status and counts.

Separately, the analytic fixture contains 144 labelled synthetic rows across four scenarios and two repetitions. A three-member ensemble is trained, checkpointed and evaluated on an unseen length/load/profile scenario. Its held-out fixture goodput MAE is {synth['heldout_goodput_mae']:.5f}, versus {synth['training_mean_baseline_goodput_mae']:.5f} for a constant training-mean predictor; it is worse on this fixture. Uncensored P99 MAE is {synth['heldout_p99_mae_s']:.2f} s. These are software diagnostics, not measurements of vLLM or evidence of hardware generalization.

![Synthetic held-out prediction diagnostic](results/synthetic/demo/model/prediction-error.png)

The synthetic replay runs default/random/GP/neural methods across three seeds with common initial information and equal admitted replay budgets. It generates goodput-versus-cost and goodput-versus-trial figures. The simulated accelerator costs are analytic fixture values; actual CPU fitting time is logged separately. No synthetic ranking is reported as a serving improvement. A CPU HTTP fixture exercises open-loop collection and parsing separately from the analytic table.

![Synthetic offline replay curves](results/synthetic/demo/replay/search-cost.png)

Focused tests cover configuration constraints, exact parser edge cases, total deadlines, missing usage, trace validation, censored tails, measurement/drain boundaries, paired cohort rules, grouped splits, budget reserves/overruns, checkpoint round trips, replay visibility and interrupted-run reconciliation. The final integrated suite passes {tests_passed} tests; its transcript is `results/software-tests.txt`. This is automated engineering review, not course-required human peer review.

## 8. Unrun experiments, failure analysis and limitations

The measured default/random/GP/neural search comparison is NOT EVALUATED. There is no reliable development baseline or independently validated feasible finalist. The held-out-load prediction experiment, live finalist validation, 1,000-completion repeated tail windows, W8A8 matched-settings ablation and fixed-compute held-out reasoning comparison are likewise unrun. The optional TP/replica, disaggregation and HCCL experiments lack the required hardware/backend capabilities.

The observed failure mechanisms are distinct: overload/deadline censoring in the initial pilot; reasoning truncation at 1,024 tokens; queueing and unfinished final output in the longer-allowance single-sequence run; and variable effective KV capacity under shared-memory pressure. The diagnostics are too small to attribute speed changes to scheduler settings because load, timeout and output allowance also differ between the first and later runs. The matched allowance pair supports only its stated development contrast.

The host is not a dedicated inference appliance. Memory compression/swap, background applications, thermal throttling and client CPU competition can affect results. The raw Prometheus samples retain engine queues/cache ratios but there is no validated device peak-memory-byte series. Process RSS is not substituted for GPU residency. Model contamination, prompt/parser sensitivity, stochastic generation, tiny sample size and winner's curse remain limitations. No portability claim extends to CUDA or Ascend devices never evaluated.

A larger allowance can permit a correct final answer but is not guaranteed to be used well. These data do not establish stronger intrinsic reasoning. Tuning amortization cannot be estimated reliably because there is no validated deployment-cost saving relative to a quality-matched reference. If such a saving is later measured, break-even answer volume is tuning accelerator-seconds divided by saved accelerator-seconds per compliant correct answer, using the same accounting boundary.

## 9. Continuation and conclusion

The next scientific step is a bounded development-only completion and service-capacity pilot with a longer measurement window and a cohort large enough to estimate quality, preserving the reference model and actual reasoning requirement. Choose output allowances, SLOs and loads using development data, then freeze them before optimizer selection and test scoring. Do not expand a nine-point search merely to produce more rows if all outcomes remain infeasible or uninformative. The existing code can then execute equal-cost searches, repeated incumbent checks and the four-arm deadline experiment.

A minimal exact next command is the `reproduce-run` example in README, using the saved allowance run instead of the short diagnostic. It needs one compatible Apple GPU/Metal runtime, the pinned 3.8 GB checkpoint, shared-memory headroom, an idle coordinated experiment queue and up to the declared approximately 575 accelerator-second reservation bound. Actual prior use was {by['allowance-metal-2048-00']['cost']['accelerator_seconds']:.2f} seconds. No paid resources are provisioned.

The implementation establishes a defensible measurement and learning pipeline and a real local counterexample to treating latency-only completion as useful reasoning. The observed results are limited to this host, checkpoint, development questions and pilot protocols. Whether neural-guided tuning beats random search or GP optimization under an equal real budget remains an open empirical question.

## Reproduction appendix and references

Use README for isolated environment setup, pinned download, exact saved-run replay, tests and fresh synthetic diagnostics. `scripts/install_metal.sh` uses the committed serving lock. `requirements-lock-macos-cpu.txt` pins the tested client/analysis environment. `scripts/build_report.py` regenerates tables/figures/text from saved evidence; `scripts/render_report_pdf.py` produces the PDF. Raw run directories and source manifests are immutable; derived summaries are regenerated explicitly.

The final freeze requires two independently quality-validated measured selections, the same frozen question cohort and two development-selected allowances. `fixed-compute` enforces that gate, counts unanswered/unoffered questions in the full denominator, and reports actual total reservations. A common service deadline does not automatically equal total compute because startup/shutdown costs may differ; unmatched costs are reported as a frontier.

Bibliographic entries are in `docs/references.bib`. The source-verification ledger includes the requested vLLM, Ascend, W8A8, disaggregation and Huawei HCCL pages, including links that returned only an unreadable site shell. Unsupported controls were not inferred from those shells. `docs/defence.md` contains the presentation outline and oral-defence questions. `docs/contributions.md` attributes the implementation and automated reviews without inventing teammate contributions or completed human review.

**Selected primary references.** Full systems link verification is supplied separately.

[1] Cheng, K., Wang, Z., Hu, W., Yang, T., Li, J., and Zhang, S. (2025). SCOOT: SLO-Oriented Performance Tuning for LLM Inference Engines. The ACM Web Conference. [Version 2 record](https://arxiv.org/abs/2408.04323v2).

[2] Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., and Zhang, H. (2024). DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. OSDI, pp. 193-210. [USENIX proceedings](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin).

[3] Snoek, J., Rippel, O., Swersky, K., Kiros, R., Satish, N., Sundaram, N., Patwary, M., Prabhat, and Adams, R. (2015). Scalable Bayesian Optimization Using Deep Neural Networks. ICML, PMLR 37, pp. 2171-2180. [Proceedings](https://proceedings.mlr.press/v37/snoek15.html).

[4] Qwen Team. Qwen3-1.7B model card and checkpoint. Apache-2.0; pinned revision stated in Section 2. [Model source](https://huggingface.co/Qwen/Qwen3-1.7B).

[5] OpenAI. GSM8K: Grade School Math repository. Pinned commit and untouched official test split stated in Section 2. [Dataset source](https://github.com/openai/grade-school-math).

[6] vLLM Project. vLLM Metal v0.32.0 release, matching upstream core and backend documentation. [Release](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0).

[7] vLLM Project. Production metrics, release 0.32.0. [Metrics definitions](https://docs.vllm.ai/en/v0.32.0/usage/metrics/).

[8] scikit-learn developers. Gaussian processes, version 1.6. [GP implementation documentation](https://scikit-learn.org/1.6/modules/gaussian_process.html).

All linked sources were accessed on 3 November 2026. Existing models, data, backend code and optimization ideas are attributed; our original work is the integration, evaluator/accounting protocol, tests and recorded local observations.

'''
(root/'REPORT.md').write_text(report)
write_json(root/'results/execution-status.json',{'status':'implementation_and_bounded_feasibility_study','measured_runs':len(rows),'measurement_request_instances':sum(r['metrics']['offered'] for r in rows),'reserved_accelerator_seconds':cost,'official_test_used':False,'measured_optimizer_comparison':'NOT EVALUATED','final_fixed_compute':'NOT EVALUATED','human_peer_review':'PENDING'})
print('Report regenerated from saved evidence.')
