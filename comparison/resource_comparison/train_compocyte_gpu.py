#!/usr/bin/env python3
"""Benchmark Compocyte v2_gpu training/inference time and memory.

Usage:
    python train_compocyte_gpu.py --subset oelen_e1_500k
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import scanpy as sc
import torch

import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.models.fit_methods import train_resource_budget
from Compocyte.core.tools import infer_levels

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "gpu")
ROOT_NODE = "Blood"


def reset_cuda_peak():
    """Reset torch's CUDA peak counters.

    WARNING: this initialises the CUDA context in the current process.  It must
    therefore never be called before Compocyte forks its training workers -- a
    forked child that inherits an initialised CUDA context cannot initialise
    CUDA again and dies with
    "RuntimeError: Cannot re-initialize CUDA in forked subprocess".
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cuda_peak_bytes():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    hierarchy_path = os.path.join(
        os.path.dirname(HERE), "data", "suco_pbmc_extracted", "hierarchy.pkl"
    )
    with open(hierarchy_path, "rb") as f:
        hierarchy = pickle.load(f)

    _, obs_names = infer_levels(
        hierarchy=hierarchy, labels="annotation", root_node=ROOT_NODE, adata=train
    )

    hc = HierarchicalClassifier(
        save_path=os.path.join(args.out_dir, args.subset, "compocyte_gpu_model"),
        adata=train,
        root_node=ROOT_NODE,
        dict_of_cell_relations=hierarchy,
        obs_names=obs_names,
    )
    hc.num_threads = args.num_threads
    fit_kwargs = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "selection_max_cells": 100_000,
    }
    nodes = [n for n in hc.graph.nodes if len(list(hc.graph.successors(n))) >= 1]

    # The training pool uses the 'fork' start method, so the CUDA context must
    # NOT be initialised in the parent before the pool is created.  Fail loudly
    # instead of letting the workers die with an opaque CUDA error.
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA is already initialised in the parent process; forked training "
            "workers would fail with 'Cannot re-initialize CUDA in forked "
            "subprocess'.  Do not call any torch.cuda.* function before "
            "hc._train_nodes_in_pool()."
        )

    # Training phase
    train_sampler = PeakMemorySampler(gpu=True)
    train_sampler.start()
    t0 = time.perf_counter()
    if args.parallel > 0:
        results = hc._train_nodes_in_pool(nodes, args.parallel, fit_kwargs=fit_kwargs)
        for node, params in zip(nodes, results):
            if params is not None:
                for key in params.keys():
                    if params.get(key) is not None:
                        hc.graph.nodes[node][key] = params.get(key)
    else:
        for node in nodes:
            params = hc.train_single_node(node, **fit_kwargs)
            if params is not None:
                for key in params.keys():
                    if params.get(key) is not None:
                        hc.graph.nodes[node][key] = params.get(key)
    t1 = time.perf_counter()
    train_peak = train_sampler.stop()
    gpu_train_peak = train_sampler.gpu_peak_bytes
    # The pool is closed at this point, so touching CUDA is safe now.
    gpu_torch_train_peak = cuda_peak_bytes()

    # Inference phase
    hc.load_adata(test)
    reset_cuda_peak()
    inf_sampler = PeakMemorySampler(gpu=True)
    inf_sampler.start()
    t2 = time.perf_counter()
    hc.predict_all_child_nodes(hc.root_node)
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()
    gpu_inf_peak = inf_sampler.gpu_peak_bytes
    gpu_torch_inf_peak = cuda_peak_bytes()

    result = {
        "method": "Compocyte (GPU)",
        "subset": args.subset,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(train.n_vars),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "parallel": args.parallel,
        "device_used": "cuda" if torch.cuda.is_available() else "cpu",
        "train_s": t1 - t0,
        "inference_s": t3 - t2,
        "rss_train_peak_bytes": int(train_peak),
        "rss_inference_peak_bytes": int(inference_peak),
        "rss_total_peak_bytes": int(max(train_peak, inference_peak)),
        "gpu_train_peak_bytes": int(gpu_train_peak),
        "gpu_inference_peak_bytes": int(gpu_inf_peak),
        "gpu_peak_bytes": int(max(gpu_train_peak, gpu_inf_peak)),
        "gpu_peak_mb": float(max(gpu_train_peak, gpu_inf_peak) / 1024 ** 2),
        "gpu_torch_peak_bytes": int(max(gpu_torch_train_peak, gpu_torch_inf_peak)),
    }
    path = write_json(result, args.out_dir)
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    sys.exit(main())
