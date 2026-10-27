#!/usr/bin/env python3
"""Optimized v2.0: large-atlas celltype annotation timing + preds.

Run with the optimized env only:
    conda run -n compo python comparison/run_v2_pretrained_timing.py [--n-cells 120000] [--parallel]
Compares against result/v1_preds.csv.gz when present (exact-match check).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
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
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--legacy-fallback", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.legacy_fallback:
        os.environ["COMPOCYTE_LEGACY_FALLBACK"] = "1"

    import torch
    from Compocyte.core.models.dense_torch import resolve_device

    atlas = build_large_atlas(args.n_cells)
    device_used = str(resolve_device(args.device))

    t0 = time.perf_counter()
    hc = til_pretrained()
    t1 = time.perf_counter()
    hc.load_adata(atlas)
    t2 = time.perf_counter()
    if torch.cuda.is_available() and device_used.startswith("cuda"):
        torch.cuda.synchronize()
    hc.predict_all_child_nodes(
        hc.root_node, device=args.device, batch_size=args.batch_size,
        parallel=args.parallel, max_workers=args.max_workers)
    if torch.cuda.is_available() and device_used.startswith("cuda"):
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    total = t3 - t0
    pred_cols = [c for c in hc.adata.obs.columns if c.endswith("_pred")]
    timing = {
        "version": "v2.0-optimized",
        "package_file": os.path.abspath(Compocyte.__file__),
        "git_sha": git_sha("/home/ritwik24222/Compocyte_walle_lab"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": device_used,
        "batch_size": args.batch_size,
        "parallel": args.parallel,
        "max_workers": args.max_workers,
        "legacy_fallback": args.legacy_fallback,
        "n_cells": int(atlas.n_obs),
        "n_genes": int(atlas.n_vars),
        "load_model_s": t1 - t0,
        "load_adata_s": t2 - t1,
        "predict_s": t3 - t2,
        "total_s": total,
        "cells_per_s": atlas.n_obs / total,
        "pred_cols": pred_cols,
    }
    with open(os.path.join(args.outdir, "v2_timing.json"), "w") as f:
        json.dump(timing, f, indent=2)

    preds = hc.adata.obs[pred_cols]
    preds.to_csv(os.path.join(args.outdir, "v2_preds.csv.gz"), compression="gzip")

    counts = {c: hc.adata.obs[c].value_counts().to_dict() for c in pred_cols}
    with open(os.path.join(args.outdir, "v2_counts.json"), "w") as f:
        json.dump(counts, f, indent=2, default=str)

    v1_path = os.path.join(args.outdir, "v1_preds.csv.gz")
    v1_time_path = os.path.join(args.outdir, "v1_timing.json")
    if os.path.exists(v1_path):
        ref = pd.read_csv(v1_path, index_col=0, compression="gzip")
        shared = [c for c in pred_cols if c in ref.columns]
        filled = preds[shared].replace("", np.nan).fillna("__NA__").values
        ref_filled = ref[shared].replace("", np.nan).fillna("__NA__").values
        match = {c: bool((filled[:, i] == ref_filled[:, i]).all())
                 for i, c in enumerate(shared)}
        rowwise = bool((filled == ref_filled).all()) if shared else None
        n_mismatch = int((filled != ref_filled).any(axis=1).sum()) if shared else None
        compare = {"shared_cols": shared, "per_col_exact_match": match,
                   "all_exact_match": rowwise, "n_mismatch_cells": n_mismatch,
                   "n_cells": int(preds.shape[0])}
        if os.path.exists(v1_time_path):
            v1t = json.load(open(v1_time_path))
            compare["speedup_total"] = v1t["total_s"] / total
            compare["speedup_predict"] = v1t["predict_s"] / (t3 - t2)
        with open(os.path.join(args.outdir, "compare_summary.json"), "w") as f:
            json.dump(compare, f, indent=2)
        print(json.dumps(compare, indent=2))

    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v
                      for k, v in timing.items() if k != "pred_cols"}, indent=2))
    print(pred_cols)


if __name__ == "__main__":
    sys.exit(main())
