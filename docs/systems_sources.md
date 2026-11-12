# Primary systems-source verification log

Retrieved 3 November 2026. Versioned documents constrain implementation; moving latest pages provide context only. No quotation or tutorial performance number is treated as our result.

| Source | Verified use in this project |
|---|---|
| [vLLM website](https://vllm.ai/) and [repository](https://github.com/vllm-project/vllm) | Project provenance and discovery of Apple Metal plugin |
| [vLLM Metal v0.32.0 release](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0) | Pinned Apple wheels, Python/macOS requirements, matching vLLM core, MLX vs MPS boundary |
| [Metal release pyproject](https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.32.0/pyproject.toml) | Dependency compatibility; installed freeze records actual resolution |
| [Metal release configuration](https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.32.0/docs/configuration.md) | Memory fraction semantics, lazy KV allocation, pressure probe, sampled MLX support |
| [Metal release supported models](https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.32.0/docs/supported_models.md) | Qwen3 family and prefix-cache support |
| [Metal release platform source](https://github.com/vllm-project/vllm-metal/blob/v0.32.0/vllm_metal/platform.py) | TP>1 rejected, chunked prefill supported, KV dtype validation; installed source also inspected |
| [v0.32.0 serve CLI](https://docs.vllm.ai/en/v0.32.0/cli/serve/) | Launch argument spelling and scheduler semantics; also checked installed help |
| [v0.32.0 bench serve CLI](https://docs.vllm.ai/en/v0.32.0/cli/bench/serve/) | Online load benchmark, latency-only SLO goodput; also checked installed help |
| [Latest bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve/) | Requested source read; not substituted for installed-release verification |
| [v0.32.0 production metrics](https://docs.vllm.ai/en/v0.32.0/usage/metrics/) | Prometheus endpoints and queue/cache/server latency telemetry; client timing remains authoritative for request SLOs |
| [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0, reasoning sampling and math answer formatting, vLLM support floor |
| [Pinned Qwen3 config](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json) | Architecture, 16 attention heads, 8 KV heads, BF16, max context 40960 |
| [vLLM-Ascend repository](https://github.com/vllm-project/vllm-ascend) | Hardware-plugin provenance; no installed Ascend stack here |
| [Requested Ascend support matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/) | Index; followed to actual feature and installation tables |
| [Ascend feature matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/feature_matrix.html) | Conditional feature compatibility, not a local support claim |
| [Ascend installation](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html) | Release-specific stack compatibility must precede NPU work |
| [Ascend quantization](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ModelSlim and LLM-Compressor formats/tools; no suitable small-model checkpoint or kernel inferred |
| [Upstream disaggregation](https://docs.vllm.ai/en/latest/features/disagg_prefill/) | Separate prefill/decode engines and connector requirement; no presumed throughput gain |
| [Ascend disaggregation design](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/disaggregated_prefill.html) | Conditional Ascend extension only |
| [Requested HCCL introduction](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_intro.md) and [buffer page](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_BUFFSIZE.md) | Retrieved site shell without technical body; not used as verified configuration evidence |
| [Readable CANN 9.0 buffer reference](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0005.html) | Buffer units and per-communicator receive/send memory; not an installed-stack recommendation |

The initially considered CUDA vLLM 0.10.2 target was superseded by the actually installed Apple stack. Archived source reading of its [serve](https://docs.vllm.ai/en/v0.10.2/cli/serve.html), [bench](https://docs.vllm.ai/en/v0.10.2/cli/bench/serve.html), and [model support](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html) is not the basis of the final runtime claim. The installed 0.32.0 help is the operational check.

## Related-work and dataset cross-check

The systems agent also read the required [SCOOT abstract/version record](https://arxiv.org/abs/2408.04323), [DistServe OSDI24 page](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin), [DNGO proceedings entry](https://proceedings.mlr.press/v37/snoek15.html), and [original GSM8K repository](https://github.com/openai/grade-school-math). The roles are distinct: SCOOT establishes prior serving autotuning, DistServe motivates SLO-aware disaggregation, DNGO establishes neural surrogate precedent, and GSM8K supplies the official reasoning data/split and answer format. None supplies a result for this implementation. See the main references/report for complete bibliographic entries.
