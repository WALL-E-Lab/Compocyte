#!/usr/bin/env python3
"""Python wrapper for CHETAH R benchmark.

Loads the deterministic subset, exports a temporary HVG-filtered MTX, calls
train_chetah.R, and cleans up the temporary files.

Usage:
    python run_chetah.py --subset oelen_e1_500k
"""
import argparse
import os
import subprocess
import sys

from shared import (
    cleanup_r_temp_dir,
    export_for_r,
    load_subset,
    make_r_temp_dir,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
RSCRIPT = "Rscript"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    tmpdir = make_r_temp_dir()
    try:
        export_for_r(train, test, tmpdir, n_top_genes=5000)
        cmd = [
            RSCRIPT,
            os.path.join(HERE, "train_chetah.R"),
            "--subset", args.subset,
            "--data-dir", tmpdir,
            "--out-dir", args.out_dir,
        ]
        subprocess.run(cmd, check=True)
    finally:
        cleanup_r_temp_dir(tmpdir)


if __name__ == "__main__":
    sys.exit(main())
