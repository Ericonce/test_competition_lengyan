#!/usr/bin/env bash
# Activate radar and keep runtime files in the writable project results directory.
set -euo pipefail
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: bash scripts/run_on_ascend.sh train/<configuration>/train.py"
    if [[ $# -eq 0 ]]; then exit 2; fi
    exit 0
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v conda >/dev/null 2>&1; then
    conda_root="$(conda info --base)"
else
    conda_root="/data/lym/anaconda3"
fi
if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
    echo "Conda not found. Activate the radar environment before running Python." >&2
    exit 1
fi
source "$conda_root/etc/profile.d/conda.sh"
conda activate radar
runtime_dir="$project_root/results/ascend_runtime"
mkdir -p "$runtime_dir/logs" "$runtime_dir/cache"
# CANN uses Unix sockets, whose paths must stay short in deeply nested workspaces.
export TMPDIR=/tmp
export ASCEND_PROCESS_LOG_PATH="$runtime_dir/logs"
export ASCEND_WORK_PATH="$runtime_dir"
export ASCEND_CACHE_PATH="$runtime_dir/cache"
export XDG_CACHE_HOME="$runtime_dir/cache"
export PULSECLS_CACHE_DIR="$runtime_dir/cache/data"
cd "$project_root"
exec python "$@"
