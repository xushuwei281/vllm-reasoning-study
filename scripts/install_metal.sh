#!/usr/bin/env bash
# Reproduce the serving environment without running the upstream global installer.
# Usage: bash scripts/install_metal.sh /absolute/path/to/project-scratch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK_WORK_DIR="${1:-$PROJECT_DIR/work}"
if [[ $# -gt 1 ]]; then
  printf '%s\n' 'Usage: bash scripts/install_metal.sh [scratch-directory]' >&2
  exit 2
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' 'This pinned environment requires Apple Silicon macOS.' >&2
  exit 2
fi
MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if [[ "$MACOS_MAJOR" -lt 15 ]]; then
  printf '%s\n' 'vLLM Metal 0.32.0 requires macOS 15 or later.' >&2
  exit 2
fi
mkdir -p "$TASK_WORK_DIR"
TASK_WORK_DIR="$(cd "$TASK_WORK_DIR" && pwd)"
export UV_CACHE_DIR="$TASK_WORK_DIR/uv-cache"
export UV_PYTHON_INSTALL_DIR="$TASK_WORK_DIR/uv-python"
export PIP_CACHE_DIR="$TASK_WORK_DIR/pip-cache"
export XDG_CACHE_HOME="$TASK_WORK_DIR/cache"
export HF_HOME="$TASK_WORK_DIR/cache/huggingface"
BOOTSTRAP_DIR="$TASK_WORK_DIR/metal-bootstrap"
RUNTIME_DIR="$TASK_WORK_DIR/metal-env"
FREEZE="$PROJECT_DIR/docs/metal-runtime.freeze.txt"
if [[ ! -f "$FREEZE" ]]; then
  printf 'Missing committed dependency freeze: %s\n' "$FREEZE" >&2
  exit 2
fi
if [[ ! -x "$BOOTSTRAP_DIR/bin/python" ]]; then
  python3 -m venv "$BOOTSTRAP_DIR"
fi
"$BOOTSTRAP_DIR/bin/python" -m pip install --disable-pip-version-check 'uv==0.9.6'
if [[ ! -x "$RUNTIME_DIR/bin/python" ]]; then
  "$BOOTSTRAP_DIR/bin/uv" venv --python 3.12.12 "$RUNTIME_DIR"
fi
"$RUNTIME_DIR/bin/python" -I -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info[:3]==(3,12,12), "Expected native Python 3.12.12; choose a fresh scratch directory"'

# The full freeze pins all dependencies and contains these inspected release wheels:
# https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0%2Bcpu-cp38-abi3-macosx_11_0_arm64.whl
# https://github.com/vllm-project/vllm-metal/releases/download/v0.32.0/vllm_metal-0.32.0-cp312-cp312-macosx_15_0_arm64.whl
# It also fixes mlx-lm to its immutable Git revision. No model weights are downloaded here.
"$BOOTSTRAP_DIR/bin/uv" pip install --python "$RUNTIME_DIR/bin/python" -r "$FREEZE"
"$BOOTSTRAP_DIR/bin/uv" pip freeze --python "$RUNTIME_DIR/bin/python" > "$TASK_WORK_DIR/metal-installed.freeze.txt"
"$RUNTIME_DIR/bin/python" -I - <<'PY'
from importlib.metadata import version
expected = {'vllm': '0.32.0+cpu', 'vllm-metal': '0.32.0', 'mlx': '0.32.2'}
for name, wanted in expected.items():
    found = version(name)
    if found != wanted:
        raise SystemExit(f'{name}: expected {wanted}, found {found}')
print('Pinned package versions verified; model execution remains a separate smoke test.')
PY
mkdir -p "$TASK_WORK_DIR/metal-help"
"$RUNTIME_DIR/bin/vllm" --version > "$TASK_WORK_DIR/metal-help/version.txt"
"$RUNTIME_DIR/bin/vllm" serve --help=all > "$TASK_WORK_DIR/metal-help/serve_help.txt"
"$RUNTIME_DIR/bin/vllm" bench serve --help > "$TASK_WORK_DIR/metal-help/bench_help.txt"
printf 'Serving executable: %s\n' "$RUNTIME_DIR/bin/vllm"
printf 'Model cache for subsequent commands: %s\n' "$HF_HOME"
printf 'Installed dependency evidence: %s\n' "$TASK_WORK_DIR/metal-installed.freeze.txt"
