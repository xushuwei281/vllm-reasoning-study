# Capability audit - 3 November 2026

## Active CUDA continuation

After the user authorized Hugging Face spending below US$10, the live study moved to one allocated NVIDIA A10G with 24,146,608,128 device-memory bytes. The pinned image is `vllm/vllm-openai:v0.10.2`; measured runtime versions are Python 3.12.11, PyTorch 2.8.0+cu128, Transformers 4.56.1, NVIDIA driver 580.178.04 and NCCL 2.27.3. The job flavor grants 12 vCPU and 46 GB RAM. Container inventory sees 48 host logical CPUs and 200,454,582,272 host RAM bytes; these host-visible totals are not a claim of exclusive entitlement. The environment manifest's generic Apple unified-memory note is inapplicable to this CUDA allocation and is preserved only as raw provenance.

| Feature | Verified CUDA evidence | Evaluated scope |
|---|---|---|
| Qwen3-1.7B reference | Pinned model/tokenizer revision; successful thinking-mode baseline with naturally completed scored answers | BF16 reference on one A10G |
| Batch/sequence controls | Installed 0.10.2 CLI help and actual successful server launches | Three batched-token limits crossed with four sequence limits |
| Chunked prefill and prefix caching | Installed enable/disable flags and saved commands | Chunked prefill on; prefix caching off throughout |
| Memory and KV dtype | Saved command and NVIDIA device-memory telemetry | Memory fraction 0.8, KV dtype auto; no low-bit KV claim |
| Streaming and server metrics | Real streamed requests, terminal token usage, raw Prometheus and NVIDIA samples | TTFT, per-request TPOT, E2E, queue and memory traces; streaming gaps do not establish per-token ITL |
| W8A8 | compressed-tensors is installed, but no exact quantized checkpoint/calibration/kernel evidence was established | **NOT EVALUATED**; package availability does not verify kernel execution or quality |
| Tensor parallelism / replicas | One allocated device; TP=1 verified | Multi-GPU TP-versus-replicas **NOT EVALUATED** |
| Disaggregated prefill/decode | No verified split deployment, additional device pool or KV-transfer smoke test | **NOT EVALUATED** |
| Ascend HCCL | No Ascend device, CANN, torch_npu or HCCL | **NOT EVALUATED**; installed NCCL is not an HCCL experiment |

The cloud help preflight uses `vllm serve --help` for 0.10.2, whose `--help=all` behavior differs from the later local Metal release. Runtime and failed-preflight artifacts are retained under `results/continuation`. Trial outcomes and final completion belong to saved measured evidence; this capability table is not a claim that every planned trial succeeded. See [cloud execution provenance](cloud_execution.md).

## Historical local Metal stage

The discovered host is an Apple M2 MacBook Air (8 CPU cores, 8 GPU cores), 16 GiB shared unified memory, macOS 27.0.1, arm64. It has one Apple GPU, no detected NVIDIA/Ascend device and no multi-device interconnect to compare. There was no vLLM installation initially. A **local isolated** vLLM Metal installation now passes CLI preflight; Apple hardware must not be described as categorically unsupported by vLLM. No global packages or machine network settings were changed.

The project targets `vllm==0.32.0+cpu` with `vllm-metal==0.32.0`, MLX 0.32.2 and Python 3.12.12. The `+cpu` wheel supplies the upstream engine; the external Metal plugin supplies Apple GPU execution. The full resolved package set is in `metal-runtime.freeze.txt`. vLLM's installed `--version`, `serve --help=all`, and `bench serve --help` were executed successfully; all flags emitted by `build_serve_command` were found. A successful help check is not proof of model execution or correct kernels. Live smoke and research-run outcomes are recorded by the root experiment runner.

| Feature | Local capability/evidence | Experiment status / restriction |
|---|---|---|
| Qwen3-1.7B reference checkpoint | Apache-2.0; immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; Qwen3 architecture supported by Metal MLX | Full model smoke/performance status belongs to saved run records; do not infer it from this audit |
| BF16 / FP16 | Model BF16, Metal text path supports reference dtypes; actual dtype is explicit in argv | Core reference experiment, subject to live smoke |
| INT8 W8A8 | No verified Qwen3-1.7B W8A8 activation/kernel path on this Metal installation | **NOT EVALUATED**; MLX weight-only 8-bit is not W8A8 |
| Tensor parallelism | Installed Metal platform explicitly rejects TP > 1 | TP=1 only; TP-versus-replicas **NOT EVALUATED**, one device |
| Chunked prefill | Installed platform explicitly enables mixed prefill/decode with paged attention; CLI verified | Fixed on in initial search; scheduler budget is not answer cap |
| `max_num_batched_tokens` | CLI verified; engine logs resolved value | Bounded search dimension; `None` omits override for default baseline |
| `max_num_seqs` | CLI verified | Engine capacity only; client offered load remains independent |
| `gpu_memory_utilization` | Passed to Metal KV planner; CLI verified | Conservative narrow range; fraction is a capacity limit, not compute utilization or pinned physical reservation |
| Prefix caching | Qwen3 Metal support documented; enable/disable flags verified | Fixed off to avoid repeated-question cache effects |
| KV cache dtype | Installed Metal validation keeps dense KV in model dtype | `auto` explicit, no 8-bit KV claim |
| `/metrics` + client streaming | Versioned vLLM documents Prometheus metrics and OpenAI streaming | Verify live metric names; retain raw text; unavailable counters stay missing |
| Prefill/decode connector | No installed/verified compatible split deployment or additional device pool | **NOT EVALUATED**; no speculative connector command |
| Ascend HCCL | No Ascend device, torch_npu, vLLM-Ascend, CANN or HCCL runtime | **NOT EVALUATED**; no HCCL variable emitted |
| Interconnect | Single Apple SoC with shared memory | No NCCL/HCCL or cross-node traffic in this scope |

The current [Metal supported-model matrix](https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.32.0/docs/supported_models.md) establishes family support; the [release notes](https://github.com/vllm-project/vllm-metal/releases/tag/v0.32.0) identify the matching upstream core. Use the MLX backend, not experimental MPS: the release's MPS path accepts greedy requests only, conflicting with the Qwen3 thinking sampling protocol.

## Memory interpretation

Unified memory is shared with the OS, client and optimizer. The 16 GiB host capacity and GPU capacity must not be added. MLX/Metal's lazy KV allocation means configured capacity differs from resident physical use; process RSS also includes non-GPU allocations. Record both metrics with their definitions, and retain OS memory-pressure observations when available. The [pinned Metal configuration documentation](https://raw.githubusercontent.com/vllm-project/vllm-metal/v0.32.0/docs/configuration.md) explains the fraction, lazy allocation and pressure probe. Do not disable the probe to make an infeasible trial appear valid.

## Conditional extensions

W8A8 requires an exact backend-specific checkpoint, weight and activation scheme, scales, excluded layers, calibration split, quantizer revision, kernel trace, and matched development answer-quality results. `validate_config` rejects an incomplete evidence record and the command builder intentionally has no universal quantization flag. The extension must add a verified backend-specific launcher before entering the search.

For Ascend, match the **installed** vLLM, vLLM-Ascend, PyTorch, torch_npu, CANN and HCCL combination to the release's support matrix. The [Ascend quantization guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) describes ModelSlim and LLM-Compressor, but its examples do not validate this small checkpoint on the present host. The [feature matrix](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/feature_matrix.html) and [installation compatibility tables](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html) must be revisited on actual NPU hardware.

HCCL is Ascend collective communication, separate from KV transfer. The supplied CANN 9.1 URLs returned only a site shell. A readable [Huawei CANN 9.0 reference](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/hcclug/hcclenvref_07_0005.html) documents `HCCL_BUFFSIZE` in MB, a default of 200, and receive/send allocations of twice that value per communicator. This is background evidence only: no 9.0/9.1 compatibility or local tuning result is asserted. A future sensitivity test must verify the installed version and count communicator memory.

A future multi-device study must compare the same total allocated device count for TP and independent replicas. A disaggregated trial requires complete proxy timing, both device pools, transfer buffers, traffic accounting, and correctness/KV-transfer smoke tests. The [upstream disaggregation documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/) and [Ascend design](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/disaggregated_prefill.html) supply architecture context, not evidence that disaggregation improves this workload.

## Reproducing the isolated runtime

From the project workspace (replace `work` only with another scratch directory):

```sh
python3 -m venv work/metal-bootstrap
work/metal-bootstrap/bin/pip install uv==0.9.6
UV_CACHE_DIR="$PWD/work/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/work/uv-python" \
  work/metal-bootstrap/bin/uv venv --python 3.12.12 work/metal-env
UV_CACHE_DIR="$PWD/work/uv-cache" work/metal-bootstrap/bin/uv pip install \
  --python work/metal-env/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl' \
  'https://github.com/vllm-project/vllm-metal/releases/download/v0.32.0/vllm_metal-0.32.0-cp312-cp312-macosx_15_0_arm64.whl'
```

For exact dependency reproduction, install the supplied freeze in that Python 3.12 environment instead of allowing fresh dependency resolution. These release-wheel URLs were obtained from the official release installer and metadata. The upstream installer itself was inspected, not executed, because its default destination is outside the project. No paid resource was provisioned during this historical local stage; the later cloud continuation has separate explicit user authorization.
