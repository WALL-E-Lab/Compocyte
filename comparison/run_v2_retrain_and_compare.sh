#!/usr/bin/env bash
# Retrain v2_cpu and v2_gpu for wang (TIL), oelen_e1 and terekhova (PBMC),
# then generate v2_cpu vs v2_gpu comparison figures.
#
# Plan: (re)run oelen_e1 v2_cpu, terekhova v2_cpu and terekhova v2_gpu with
# the fixed code, keep wang (both) and oelen_e1 v2_gpu as-is, then compare.
# Completed steps are skipped automatically. Override with FORCE=1.
set -euo pipefail

cd "$(dirname "$0")/.."
FORCE="${FORCE:-0}"

have_outputs() {
    [ -f "$1/timing.json" ] && [ -f "$1/preds.csv.gz" ]
}

run_step() {
    local label="$1"; local outdir="$2"; shift 2
    if [ "$FORCE" = "1" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label (FORCE=1, rerunning)..."
    elif have_outputs "$outdir"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP $label (outputs exist in $outdir)."
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label..."
    fi
    "$@"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting v2_cpu/v2_gpu retraining..."

# --- TIL wang (DONE, skipped) -------------------------------------------------
run_step "Training wang TIL v2_cpu" "comparison/result/pretraining/v2_cpu/wang" \
    conda run -n compo python -u comparison/run_train_v2cpu.py \
    --dataset wang --epochs 100 --parallel 4 --batch-size 64

run_step "Training wang TIL v2_gpu" "comparison/result/pretraining/v2_gpu/wang" \
    conda run -n compo-gpu python -u comparison/run_train_v2gpu.py \
    --dataset wang --epochs 100 --batch-size 1024

# --- PBMC oelen_e1 ------------------------------------------------------------
# v2_cpu is (re)run with the fixed code; v2_gpu is kept as-is (skipped).
run_step "Training oelen_e1 PBMC v2_cpu" "comparison/result/pretraining_pbmc/v2_cpu/oelen_e1" \
    conda run -n compo python -u comparison/run_train_pbmc_v2cpu.py \
    --dataset oelen_e1 --epochs 100 --parallel 4 --batch-size 64

run_step "Training oelen_e1 PBMC v2_gpu" "comparison/result/pretraining_pbmc/v2_gpu/oelen_e1" \
    conda run -n compo-gpu python -u comparison/run_train_pbmc_v2gpu.py \
    --dataset oelen_e1 --epochs 100 --batch-size 1024

# --- PBMC terekhova (both, with fixed code) -----------------------------------
# Memory-aware scheduler inside _train_nodes_in_pool runs the giant root node
# sequentially and parallelizes the smaller nodes automatically.
run_step "Training terekhova PBMC v2_cpu" "comparison/result/pretraining_pbmc/v2_cpu/terekhova" \
    conda run -n compo python -u comparison/run_train_pbmc_v2cpu.py \
    --dataset terekhova --epochs 100 --parallel 4 --batch-size 64

run_step "Training terekhova PBMC v2_gpu" "comparison/result/pretraining_pbmc/v2_gpu/terekhova" \
    conda run -n compo-gpu python -u comparison/run_train_pbmc_v2gpu.py \
    --dataset terekhova --epochs 100 --batch-size 1024

# --- Comparison figures -------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Generating comparison figures..."
conda run -n compo python -u comparison/compare_v2_cpu_vs_gpu.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
