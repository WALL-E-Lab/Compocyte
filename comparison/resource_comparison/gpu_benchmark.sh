#!/usr/bin/env bash
# Standalone GPU compute-resource benchmark for the 5 methods that support GPU
# execution: Compocyte v2_gpu, scANVI, CellTypist, treeArches, scPoli.
#
# Results are written to comparison/resource_comparison/results/gpu/ so they
# coexist with the CPU benchmark results in results/cpu/.
#
# NOTE: CellTypist GPU requires the RAPIDS packages `cuml` and
# `rapids_singlecell` in the `scrna` environment.  The script checks for them
# before starting.  If they are missing, install them first, for example:
#     conda run -n scrna conda install -c rapidsai -c conda-forge -c nvidia \
#         cuml rapids-singlecell
# (match the RAPIDS CUDA version to your driver; this node has driver
# 525.105.17 / CUDA 12.0).
#
# Usage:
#     cd comparison/resource_comparison
#     nohup ./gpu_benchmark.sh > gpu_benchmark.log 2>&1 &
#
# NOTE: write the log to the script directory (not into results/gpu/), because
# the script may clear stale non-JSON files in results/gpu/ at startup.

set -euo pipefail

# ----------------------------- Configuration ---------------------------------
CONDA_BASE="${HOME}/miniconda3"

# Conda environments
COMP_GPU_ENV="compo-gpu"    # Compocyte v2_gpu
SOTA_ENV="scrna"            # scANVI / CellTypist / treeArches / scPoli / figure

# Subsets to benchmark (must match keys in shared.py)
SUBSETS=("liu_100k" "zhang_250k" "oelen_e1_500k")

# Minimum free GB required on the project filesystem
MIN_FREE_GB=20

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# We do NOT delete existing JSON results in results/gpu/ because Compocyte GPU
# has already been run and we want to keep those files.  Only stale non-JSON
# artefacts (model checkpoints, temporary directories, etc.) are removed.
mkdir -p results/gpu figs
find results/gpu -mindepth 1 -maxdepth 1 ! -name '*.json' -exec rm -rf {} +

# ----------------------------- Helpers ---------------------------------------
source_conda() {
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck source=/dev/null
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
        exit 1
    fi
}

run_py() {
    local env="$1"
    shift
    source_conda
    conda run -n "$env" python "$@"
}

# ----------------------------- Space check -----------------------------------
FREE_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "ERROR: only ${FREE_GB} GB free; need at least ${MIN_FREE_GB} GB." >&2
    exit 1
fi
echo "Free disk space: ${FREE_GB} GB"

# ----------------------------- Environment checks ----------------------------
echo ""
echo "===== Checking Python environments ====="
source_conda
for env in "$COMP_GPU_ENV" "$SOTA_ENV"; do
    if ! conda info --envs | awk '{print $1}' | grep -qx "$env"; then
        echo "ERROR: conda environment '$env' not found." >&2
        exit 1
    fi
done

# Compocyte GPU env needs Compocyte + working CUDA
conda run -n "$COMP_GPU_ENV" python -c "import Compocyte, torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "ERROR: Compocyte/torch(CUDA) not available in $COMP_GPU_ENV" >&2
    exit 1
}

# SOTA env needs the full stack and a working CUDA driver for the 4 GPU methods
SOTA_CHECK=$(conda run -n "$SOTA_ENV" python -c "
for pkg in ['scanpy','skmisc','scvi','celltypist','scarches','scHPL','networkx','sklearn','matplotlib','pandas','numpy','torch']:
    __import__(pkg)
import torch
assert torch.cuda.is_available(), 'CUDA not available in scrna env'
print('OK')
" 2>&1) || true
if ! echo "$SOTA_CHECK" | grep -q "OK"; then
    echo "ERROR: SOTA Python packages missing or CUDA not available" >&2
    echo "$SOTA_CHECK" >&2
    exit 1
fi

echo ""
echo "===== Checking CellTypist GPU dependencies ====="
CT_CHECK=$(conda run -n "$SOTA_ENV" python -c "
import importlib.util
missing = [p for p in ('cuml', 'rapids_singlecell') if importlib.util.find_spec(p) is None]
if missing:
    raise ImportError(f'Missing CellTypist GPU packages: {missing}')
print('OK')
" 2>&1) || true
if echo "$CT_CHECK" | grep -q "OK"; then
    RUN_CELLTYPIST_GPU=1
else
    RUN_CELLTYPIST_GPU=0
    echo "WARNING: CellTypist GPU dependencies missing (cuml/rapids_singlecell)." >&2
    echo "         Skipping CellTypist GPU. Install them and re-run to include it." >&2
    echo "$CT_CHECK" >&2
fi

# ----------------------------- Benchmark run ---------------------------------
echo ""
echo "===== Starting GPU benchmark ====="
START=$(date +%Y-%m-%d\ %H:%M:%S)
echo "Start time: $START"

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=== Subset: $subset ==="

    # Uncomment the next two lines if you want to re-run Compocyte GPU.
    # echo "  Compocyte v2_gpu ..."
    # run_py "$COMP_GPU_ENV" train_compocyte_gpu.py --subset "$subset" --epochs 100 --out-dir results/gpu || true

    echo "  scANVI (GPU) ..."
    run_py "$SOTA_ENV" train_scanvi_gpu.py --subset "$subset" --epochs 100 --out-dir results/gpu || true

    if [ "$RUN_CELLTYPIST_GPU" -eq 1 ]; then
        echo "  CellTypist (GPU) ..."
        run_py "$SOTA_ENV" train_celltypist_gpu.py --subset "$subset" --out-dir results/gpu || true
    fi

    echo "  treeArches (GPU) ..."
    run_py "$SOTA_ENV" train_treearches_gpu.py --subset "$subset" --epochs 100 --out-dir results/gpu || true

    echo "  scPoli (GPU) ..."
    run_py "$SOTA_ENV" train_scpoli_gpu.py --subset "$subset" --epochs 100 --out-dir results/gpu || true
done

echo ""
echo "===== Generating figures ====="
run_py "$SOTA_ENV" generate_figure.py --results-dir results/gpu --tag gpu

END=$(date +%Y-%m-%d\ %H:%M:%S)
echo ""
echo "===== GPU benchmark complete ====="
echo "Start: $START"
echo "End:   $END"
echo "Results: $SCRIPT_DIR/results/gpu"
echo "Figures: $SCRIPT_DIR/figs"
