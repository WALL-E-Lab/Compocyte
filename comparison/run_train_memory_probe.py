#!/usr/bin/env python3
"""Peak-RSS probe for the training data path on a large synthetic sparse matrix.

v1.0 materializes the full dense matrix (robust_scale -> torch tensor).
v2.0 keeps the matrix sparse and densifies per mini-batch.

    conda run -n compo-v1 python comparison/run_train_memory_probe.py
    conda run -n compo    python comparison/run_train_memory_probe.py
"""
import argparse
import json
import os
import resource
import sys

import numpy as np
from scipy import sparse

import Compocyte

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "result", "pretraining")


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def cur_rss():
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def synthetic(n_cells, n_genes, density, seed=0):
    rng = np.random.default_rng(seed)
    nnz = int(n_cells * n_genes * density)
    rows = rng.integers(0, n_cells, nnz)
    cols = rng.integers(0, n_genes, nnz)
    vals = rng.random(nnz).astype(np.float32) * 10
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_genes))


def dense_path(x, batch_size):
    import torch
    from sklearn.preprocessing import robust_scale
    if sparse.issparse(x):
        x = np.asarray(x.toarray())
    x = robust_scale(x, axis=1, with_centering=False, copy=False, unit_variance=True)
    t = torch.from_numpy(np.asarray(x)).to(torch.float32)
    return t.shape, t.dtype


def sparse_path(x, batch_size):
    from Compocyte.core.models.fit_methods import (
        SparseBatchDataset, _sparse_collate, row_robust_scale_sparse)
    from functools import partial
    import torch
    x = row_robust_scale_sparse(x)
    if isinstance(x, sparse.csr_matrix) or sparse.issparse(x):
        ds = SparseBatchDataset(x, np.zeros((x.shape[0], 2), dtype=np.float32))
        collate = partial(_sparse_collate, X_csr=ds.X_csr,
                          y_cat=np.zeros((x.shape[0], 2), dtype=np.float32))
        seen = 0
        for start in range(0, len(ds), batch_size):
            idx = list(range(start, min(start + batch_size, len(ds))))
            xb, _ = collate(idx)
            seen += xb.shape[0]
        return (seen, x.shape[1]), x.dtype
    return dense_path(x, batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "dense", "sparse", "baseline"],
                    default="both")
    ap.add_argument("--n-cells", type=int, default=200000)
    ap.add_argument("--n-genes", type=int, default=1011)
    ap.add_argument("--density", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch

    x = synthetic(args.n_cells, args.n_genes, args.density)
    sparse_bytes = x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    out = {
        "package_file": os.path.abspath(Compocyte.__file__),
        "torch": torch.__version__,
        "n_cells": args.n_cells,
        "n_genes": args.n_genes,
        "density": args.density,
        "csr_bytes": int(sparse_bytes),
        "dense_equivalent_bytes": int(args.n_cells * args.n_genes * 4),
        "batch_size": args.batch_size,
    }
    try:
        from Compocyte.core.models.fit_methods import train_resource_budget
        out["resource_budget"] = train_resource_budget()
    except ImportError:
        out["resource_budget"] = None

    if args.mode == "both":
        # Peak RSS is a per-process high-water mark, so each path is measured in
        # its own subprocess.
        import subprocess
        for mode in ("baseline", "dense", "sparse"):
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--mode", mode, "--n-cells", str(args.n_cells),
                   "--n-genes", str(args.n_genes), "--density", str(args.density),
                   "--batch-size", str(args.batch_size), "--outdir", args.outdir]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
                child = json.loads(proc.stdout)
                out[f"{mode}_path_peak_rss_bytes"] = child.get("peak_rss_bytes")
                out[f"{mode}_path_available"] = child.get("path_available", True)
            else:
                out[f"{mode}_path_peak_rss_bytes"] = None
                out[f"{mode}_path_available"] = False
                out[f"{mode}_path_error"] = proc.stderr.strip().splitlines()[-1:] or ""
    else:
        if args.mode == "baseline":
            pass
        else:
            path = dense_path if args.mode == "dense" else sparse_path
            try:
                path(x, args.batch_size)
            except Exception as exc:
                out["path_available"] = False
                out["path_error"] = str(exc)
            else:
                out["path_available"] = True
        out["peak_rss_bytes"] = int(rss())
        print(json.dumps(out))
        return

    tag = "v1" if "Compocyte_v1.0" in os.path.abspath(Compocyte.__file__) else "v2"
    with open(os.path.join(args.outdir, f"memory_probe_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
