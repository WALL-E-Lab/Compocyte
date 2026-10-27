#!/usr/bin/env python3
"""Benchmark CellTypist training/inference time and memory.

Usage:
    python train_celltypist.py --subset oelen_e1_500k
"""
import argparse
import json
import os
import sys
import time

import celltypist
import numpy as np
import scanpy as sc

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    train.obs["label"] = train.obs["annotation"].astype(str)
    test.obs["label"] = test.obs["annotation"].astype(str)

    sc.pp.highly_variable_genes(train, inplace=True, flavor="seurat_v3", n_top_genes=5000)
    train = train[:, train.var["highly_variable"]].copy()
    # Same HVG set for the test split
    test = test[:, train.var_names].copy()

    sc.pp.normalize_total(train, target_sum=10000)
    sc.pp.log1p(train)
    sc.pp.normalize_total(test, target_sum=10000)
    sc.pp.log1p(test)

    # Training phase
    train_sampler = PeakMemorySampler()
    train_sampler.start()
    t0 = time.perf_counter()
    model = celltypist.train(train, labels="label", feature_selection=True, with_mean=False)
    t1 = time.perf_counter()
    train_peak = train_sampler.stop()

    # Inference phase
    # celltypist 1.7.1 + scipy 1.13.1 produces a COO matrix from sparse arithmetic,
    # which does not support item assignment. Passing a dense AnnData works around this.
    test_dense = sc.AnnData(X=test.X.toarray() if hasattr(test.X, "toarray") else np.asarray(test.X),
                            obs=test.obs.copy(), var=test.var.copy())
    inf_sampler = PeakMemorySampler()
    inf_sampler.start()
    t2 = time.perf_counter()
    predictions = celltypist.annotate(test_dense, model=model, majority_voting=True)
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()

    result = {
        "method": "CellTypist",
        "subset": args.subset,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(train.n_vars),
        "train_s": t1 - t0,
        "inference_s": t3 - t2,
        "rss_train_peak_bytes": int(train_peak),
        "rss_inference_peak_bytes": int(inference_peak),
        "rss_total_peak_bytes": int(max(train_peak, inference_peak)),
    }
    path = write_json(result, args.out_dir)
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    sys.exit(main())
