#!/usr/bin/env bash
# Standalone compute-resource benchmark for 8 cell-type classifiers.
#
# Methods: Compocyte v2_cpu, scANVI (CPU), CellTypist (CPU), treeArches (CPU),
#          scPoli (CPU), CHETAH, transferData.
#
# Each method loads the source Suco PBMC h5ad and deterministically subsets
# 100k/250k/500k cells with seed=42, so every method trains on the same cells.
# R-based methods receive a temporary HVG-filtered, gzip-compressed sparse MTX.
#
# IMPORTANT setup step for treeArches:
#   The `scrna` env needs `pooch` because treeArches imports Compocyte from
#   the repository src/ tree.  Run once before the benchmark:
#       conda run -n scrna pip install pooch
#
# Usage:
#     cd comparison/resource_comparison
#     nohup ./run_benchmark.sh > benchmark.log 2>&1 &
#
# NOTE: write the log to the script directory (not into results/), because the
# script clears results/ at startup.

set -euo pipefail

# ----------------------------- Configuration ---------------------------------
CONDA_BASE="${HOME}/miniconda3"

# Conda environments
COMP_CPU_ENV="compo"        # Compocyte v2_cpu (CPU-only)
SOTA_ENV="scrna"            # scANVI / CellTypist / treeArches / scPoli / figure

# Rscript binary
RSCRIPT="Rscript"

# Subsets to benchmark (must match keys in shared.py)
SUBSETS=("liu_100k" "zhang_250k" "oelen_e1_500k")

# Minimum free GB required on the project filesystem
MIN_FREE_GB=20

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clean previous CPU results before producing new ones.
# Results are split into results/cpu/ and results/gpu/ so the two benchmarks
# can be run independently.  We do NOT delete figs/ because CPU and GPU
# figures now have distinct filenames (GPU figures carry a '_gpu' tag).
mkdir -p results/cpu results/gpu figs
find results/cpu -mindepth 1 -maxdepth 1 -exec rm -rf {} +
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
    echo "Delete comparison/data/suco_pbmc/Suco_PBMC_1.3.tar.xz or other large unused files." >&2
    exit 1
fi
echo "Free disk space: ${FREE_GB} GB"

# ----------------------------- Environment checks ----------------------------
echo ""
echo "===== Checking Python environments ====="
source_conda
for env in "$COMP_CPU_ENV" "$SOTA_ENV"; do
    if ! conda info --envs | awk '{print $1}' | grep -qx "$env"; then
        echo "ERROR: conda environment '$env' not found." >&2
        exit 1
    fi
done

# Compocyte CPU env needs Compocyte + torch
conda run -n "$COMP_CPU_ENV" python -c "import Compocyte, torch" 2>/dev/null || {
    echo "ERROR: Compocyte/torch not available in $COMP_CPU_ENV" >&2
    exit 1
}

# SOTA env needs the full stack
conda run -n "$SOTA_ENV" python - <<'PY' || { echo "ERROR: SOTA Python packages missing" >&2; exit 1; }
for pkg in ['scanpy','skmisc','scvi','celltypist','scarches','scHPL','networkx','sklearn','matplotlib','pandas','numpy']:
    __import__(pkg)
PY

echo ""
echo "===== Checking R packages ====="
"$RSCRIPT" --vanilla -e '
.libPaths(c("/home/ritwik24222/R/library", .libPaths()))
for (p in c("CHETAH","scuttle","scran","optparse","mclust","Seurat","Matrix","Rtsne","data.table","jsonlite")) {
  if (!requireNamespace(p, quietly=TRUE)) stop(paste(p, "not installed"))
}
cat("R packages OK\n")
' 2>/dev/null || {
    echo "ERROR: required R packages missing." >&2
    echo "Run in R: BiocManager::install(c('CHETAH','scuttle','scran')); install.packages(c('optparse','mclust','jsonlite'))" >&2
    exit 1
}

# ----------------------------- Benchmark run ---------------------------------
echo ""
echo "===== Starting benchmark ====="
START=$(date +%Y-%m-%d\ %H:%M:%S)
echo "Start time: $START"

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=== Subset: $subset ==="

    echo "  Compocyte v2_cpu ..."
    run_py "$COMP_CPU_ENV" train_compocyte_cpu.py --subset "$subset" --epochs 100 --out-dir results/cpu || true

    echo "  scANVI (CPU) ..."
    run_py "$SOTA_ENV" train_scanvi.py --subset "$subset" --epochs 100 --out-dir results/cpu || true

    echo "  CellTypist (CPU) ..."
    run_py "$SOTA_ENV" train_celltypist.py --subset "$subset" --out-dir results/cpu || true

    echo "  treeArches (CPU) ..."
    run_py "$SOTA_ENV" train_treearches.py --subset "$subset" --epochs 100 --out-dir results/cpu || true

    echo "  scPoli (CPU) ..."
    run_py "$SOTA_ENV" train_scpoli.py --subset "$subset" --epochs 100 --out-dir results/cpu || true

    echo "  CHETAH ..."
    run_py "$SOTA_ENV" run_chetah.py --subset "$subset" --out-dir results/cpu || true

    echo "  transferData ..."
    run_py "$SOTA_ENV" run_transferdata.py --subset "$subset" --out-dir results/cpu || true
done

echo ""
echo "===== Generating figures ====="
run_py "$SOTA_ENV" generate_figure.py --results-dir results/cpu

END=$(date +%Y-%m-%d\ %H:%M:%S)
echo ""
echo "===== Benchmark complete ====="
echo "Start: $START"
echo "End:   $END"
echo "Results: $SCRIPT_DIR/results"
echo "Figures: $SCRIPT_DIR/figs"
