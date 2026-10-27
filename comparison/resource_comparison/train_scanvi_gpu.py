#!/usr/bin/env python3
"""Benchmark GPU scANVI training/inference time and memory.

Usage:
    python train_scanvi_gpu.py --subset oelen_e1_500k
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scanpy as sc
import scvi
import torch

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "gpu")

# Lightning / PyTorch training arguments that force the model onto the first
# available CUDA device.  `accelerator="gpu"` + `devices=1` is the stable
# scvi-tools API for single-GPU training.
GPU_TRAIN_KWARGS = {"accelerator": "gpu", "devices": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment; cannot run scANVI GPU benchmark.")

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    scvi.settings.seed = args.seed
    np.random.seed(args.seed)

    train.obs["label"] = train.obs["annotation"].astype(str)
    train.obs["batch"] = train.obs["dataset"].astype(str)
    test.obs["label"] = test.obs["annotation"].astype(str)
    test.obs["batch"] = test.obs["dataset"].astype(str)

    combined = sc.concat([train, test], label="split", keys=["train", "test"])
    combined.obs["label"] = combined.obs["label"].astype(str)
    combined.obs["batch"] = combined.obs["batch"].astype(str)

    sc.pp.highly_variable_genes(combined, inplace=True, flavor="seurat_v3", n_top_genes=5000)
    combined = combined[:, combined.var["highly_variable"]].copy()

    # Training phase
    train_sampler = PeakMemorySampler(gpu=True)
    train_sampler.start()
    t0 = time.perf_counter()
    scvi.model.SCVI.setup_anndata(combined, batch_key="batch")
    arches_params = dict(
        use_layer_norm="both",
        use_batch_norm="none",
        encode_covariates=True,
        dropout_rate=0.2,
        n_layers=2,
    )
    vae = scvi.model.SCVI(combined, **arches_params)
    vae.train(max_epochs=args.epochs, **GPU_TRAIN_KWARGS)
    scanvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category="Unknown",
        labels_key="label",
        adata=combined,
    )
    scanvae.train(max_epochs=args.epochs, **GPU_TRAIN_KWARGS)
    t1 = time.perf_counter()
    train_peak = train_sampler.stop()
    gpu_train_peak = train_sampler.gpu_peak_bytes

    # Inference phase
    test_adata = combined[combined.obs["split"] == "test"].copy()
    inf_sampler = PeakMemorySampler(gpu=True)
    inf_sampler.start()
    t2 = time.perf_counter()
    scanvae_query = scvi.model.SCANVI.load_query_data(test_adata, scanvae)
    scanvae_query.train(
        max_epochs=args.epochs,
        plan_kwargs={"weight_decay": 0.0},
        check_val_every_n_epoch=10,
        **GPU_TRAIN_KWARGS,
    )
    test_adata.obs["label_pred"] = scanvae_query.predict()
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()
    gpu_inf_peak = inf_sampler.gpu_peak_bytes

    result = {
        "method": "scANVI",
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
