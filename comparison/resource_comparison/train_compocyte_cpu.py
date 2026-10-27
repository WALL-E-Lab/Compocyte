#!/usr/bin/env python3
"""Benchmark Compocyte v2_cpu training/inference time and memory.

Usage:
    python train_compocyte_cpu.py --subset oelen_e1_500k
"""
import argparse
import json
import os
import pickle
import sys
import time
import warnings

# The CPU-only conda env ("compo") ships a torch build compiled against a newer
# CUDA runtime than the installed driver supports.  Compocyte calls
# torch.cuda.is_available() while resolving the device, which emits a scary
# "CUDA initialization: The NVIDIA driver on your system is too old" warning.
# This benchmark runs on the CPU on purpose, so silence it.
warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
)

import numpy as np
import scanpy as sc
import torch

import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.models.fit_methods import train_resource_budget
from Compocyte.core.tools import infer_levels

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
ROOT_NODE = "Blood"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
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
        save_path=os.path.join(args.out_dir, args.subset, "compocyte_cpu_model"),
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

    # Training phase
    train_sampler = PeakMemorySampler()
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

    # Inference phase
    hc.load_adata(test)
    inf_sampler = PeakMemorySampler()
    inf_sampler.start()
    t2 = time.perf_counter()
    hc.predict_all_child_nodes(hc.root_node)
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()

    result = {
        "method": "Compocyte (CPU)",
        "subset": args.subset,
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(train.n_vars),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "parallel": args.parallel,
        "device_used": "cpu",
        "train_s": t1 - t0,
        "inference_s": t3 - t2,
        "rss_train_peak_bytes": int(train_peak),
        "rss_inference_peak_bytes": int(inference_peak),
        "rss_total_peak_bytes": int(max(train_peak, inference_peak)),
        "gpu_peak_bytes": 0,
        "gpu_peak_mb": 0.0,
    }
    path = write_json(result, args.out_dir)
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    sys.exit(main())
