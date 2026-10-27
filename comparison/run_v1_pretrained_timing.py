#!/usr/bin/env python3
"""Frozen v1.0 baseline: large-atlas celltype annotation timing + preds.

Run with the frozen env only:
    conda run -n compo-v1 python comparison/run_v1_pretrained_timing.py [--n-cells 120000]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import scanpy as sc
from scipy import sparse

import Compocyte
from Compocyte.data import sample_data
from Compocyte.pretrained import til_pretrained

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "result")


def build_large_atlas(target):
    base = sample_data()
    reps = int(np.ceil(target / base.n_obs))
    X = sparse.vstack([base.X] * reps, format="csr")[:target]
    idx = np.tile(np.arange(base.n_obs), reps)[:target]
    obs = base.obs.iloc[idx].copy()
    obs.index = [f"large_cell_{i:06d}" for i in range(X.shape[0])]
    return sc.AnnData(X=X, obs=obs, var=base.var.copy())


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=120000)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    atlas = build_large_atlas(args.n_cells)

    t0 = time.perf_counter()
    hc = til_pretrained()
    t1 = time.perf_counter()
    hc.load_adata(atlas)
    t2 = time.perf_counter()
    hc.predict_all_child_nodes(hc.root_node)
    t3 = time.perf_counter()

    total = t3 - t0
    pred_cols = [c for c in hc.adata.obs.columns if c.endswith("_pred")]
    timing = {
        "version": "v1.0-frozen",
        "package_file": os.path.abspath(Compocyte.__file__),
        "git_sha": git_sha(os.path.dirname(os.path.abspath(Compocyte.__file__))),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu",
        "n_cells": int(atlas.n_obs),
        "n_genes": int(atlas.n_vars),
        "load_model_s": t1 - t0,
        "load_adata_s": t2 - t1,
        "predict_s": t3 - t2,
        "total_s": total,
        "cells_per_s": atlas.n_obs / total,
        "pred_cols": pred_cols,
    }
    with open(os.path.join(args.outdir, "v1_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    preds = hc.adata.obs[pred_cols]
    preds.to_csv(os.path.join(args.outdir, "v1_preds.csv.gz"), compression="gzip")

    counts = {c: hc.adata.obs[c].value_counts().to_dict() for c in pred_cols}
    with open(os.path.join(args.outdir, "v1_counts.json"), "w") as f:
        json.dump(counts, f, indent=2, default=str)

    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v
                      for k, v in timing.items() if k != "pred_cols"}, indent=2))
    print(pred_cols)


if __name__ == "__main__":
    sys.exit(main())
