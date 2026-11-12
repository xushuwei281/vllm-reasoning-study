#!/usr/bin/env bash
# Reproduce the saved 2,048-token development diagnostic under a new run ID.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scratch_dir="${1:-$project_dir/work}"
scratch_dir="$(cd -- "$scratch_dir" && pwd)"
analysis_python="$scratch_dir/analysis-env/bin/python"
if [[ ! -x "$analysis_python" ]]; then analysis_python="$scratch_dir/.venv/bin/python"; fi
model_path_file="$scratch_dir/model-path.txt"
if [[ ! -f "$model_path_file" ]]; then model_path_file="$scratch_dir/qwen-snapshot-path.txt"; fi
[[ -x "$analysis_python" && -x "$scratch_dir/metal-env/bin/vllm" && -f "$model_path_file" ]] || {
  echo 'Prepare analysis/Metal environments and the pinned model using README first.' >&2; exit 2;
}
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
"$analysis_python" -m vllm_tuner preflight --vllm "$scratch_dir/metal-env/bin/vllm" --version 0.32.0 --output "$scratch_dir/reproduction-preflight"
"$scratch_dir/metal-env/bin/python" -m vllm_tuner environment --output "$scratch_dir/reproduction-environment/metal.json"
"$analysis_python" -m vllm_tuner reproduce-run \
  --study "$project_dir/configs/allowance-metal.json" \
  --original-run "$project_dir/results/measured/runs/allowance-metal-2048-00" \
  --data "$scratch_dir/data" --tokenizer-path "$(cat "$model_path_file")" \
  --environment "$scratch_dir/reproduction-environment/metal.json" \
  --capabilities "$scratch_dir/reproduction-environment/capabilities.json" \
  --vllm "$scratch_dir/metal-env/bin/vllm" \
  --hf-home "$scratch_dir/cache/huggingface" \
  --ledger "$scratch_dir/reproduction/budget.jsonl" \
  --output "$scratch_dir/reproduction" --run-id "allowance-repeat-$(date -u +%Y%m%dT%H%M%SZ)"
