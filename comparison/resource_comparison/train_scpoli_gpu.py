#!/usr/bin/env python3
"""Benchmark GPU scPoli training/inference time and memory.

Usage:
    python train_scpoli_gpu.py --subset oelen_e1_500k

scPoli's trainer automatically selects CUDA when `torch.cuda.is_available()`.
This script additionally samples device-level GPU memory with nvidia-smi so
that the benchmark is comparable to the other GPU methods.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scanpy as sc
import torch
from scarches.models.scpoli import scPoli

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "gpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment; cannot run scPoli GPU benchmark.")

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    train.obs["last_label"] = train.obs["annotation"].astype(str)
    train.obs["batch"] = train.obs["dataset"].astype(str)
    test.obs["last_label"] = test.obs["annotation"].astype(str)
    test.obs["batch"] = test.obs["dataset"].astype(str)

    combined = sc.concat([train, test], label="split", keys=["train", "test"])
    combined = sc.AnnData(
        X=combined.X.astype(np.float32), obs=combined.obs, var=combined.var
    )
    combined.obs["last_label"] = combined.obs["last_label"].astype(str)
    combined.obs["batch"] = combined.obs["batch"].astype(str)

    sc.pp.highly_variable_genes(combined, inplace=True, flavor="seurat_v3", n_top_genes=3000)
    combined = combined[:, combined.var["highly_variable"]].copy()

    condition_key = "batch"
    cell_type_key = ["last_label"]
    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    # Indices of the labeled (train) cells within the combined object
    train_idx = np.where(combined.obs["split"] == "train")[0].tolist()
    test_idx = np.where(combined.obs["split"] == "test")[0].tolist()

    # Training phase
    train_sampler = PeakMemorySampler(gpu=True)
    train_sampler.start()
    t0 = time.perf_counter()
    scpoli_model = scPoli(
        adata=combined,
        condition_keys=condition_key,
        cell_type_keys=cell_type_key,
        embedding_dims=3,
    )
    # Do NOT set pretraining_epochs == n_epochs: scPoliTrainer only initializes
    # prototypes when self.epoch == self.pretraining_epochs, and epochs are
    # 0-indexed.  Using the default (floor(0.9 * n_epochs)) is safe.
    scpoli_model.train(
        n_epochs=args.epochs,
        early_stopping_kwargs=early_stopping_kwargs,
        eta=5,
    )
    # Sanity check that the model actually landed on the GPU
    device = next(scpoli_model.model.parameters()).device
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"scPoli model is on {device}, expected cuda.")
    t1 = time.perf_counter()
    train_peak = train_sampler.stop()
    gpu_train_peak = train_sampler.gpu_peak_bytes

    # Inference phase
    inf_sampler = PeakMemorySampler(gpu=True)
    inf_sampler.start()
    t2 = time.perf_counter()
    scpoli_query = scPoli.load_query_data(
        adata=combined,
        reference_model=scpoli_model,
        labeled_indices=train_idx,
    )
    scpoli_query.train(
        n_epochs=args.epochs,
        eta=10,
    )
    results_dict = scpoli_query.classify(combined[test_idx].copy(), scale_uncertainties=True)
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()
    gpu_inf_peak = inf_sampler.gpu_peak_bytes

    result = {
        "method": "scPoli",
        "subset": args.subset,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(combined.n_vars),
        "epochs": args.epochs,
        "device_used": "cuda",
        "train_s": t1 - t0,
        "inference_s": t3 - t2,
        "rss_train_peak_bytes": int(train_peak),
        "rss_inference_peak_bytes": int(inference_peak),
        "rss_total_peak_bytes": int(max(train_peak, inference_peak)),
        "gpu_train_peak_bytes": int(gpu_train_peak),
        "gpu_inference_peak_bytes": int(gpu_inf_peak),
        "gpu_peak_bytes": int(max(gpu_train_peak, gpu_inf_peak)),
        "gpu_peak_mb": float(max(gpu_train_peak, gpu_inf_peak) / 1024 ** 2),
    }
    path = write_json(result, args.out_dir)
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    sys.exit(main())
