#!/usr/bin/env bash
# Run a train/infer entry using the server's radar_env and project-local runtime files.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /data/lym/radar_env/bin/activate
runtime_dir="$project_root/results/ascend_runtime"
mkdir -p "$runtime_dir/tmp" "$runtime_dir/logs" "$runtime_dir/cache"
export TMPDIR="$runtime_dir/tmp"
export ASCEND_PROCESS_LOG_PATH="$runtime_dir/logs"
export ASCEND_WORK_PATH="$runtime_dir"
export ASCEND_CACHE_PATH="$runtime_dir/cache"
export XDG_CACHE_HOME="$runtime_dir/cache"
cd "$project_root"
if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/run_on_ascend.sh train/<configuration>/train.py" >&2
    exit 2
fi
exec python "$@"
