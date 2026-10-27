#!/usr/bin/env python3
"""Optimized v2.0 pretraining benchmark on a scRNA-seq dataset (GSE216280-like PBMC).

Same data, hierarchy and epochs as run_v1_pretraining_benchmark.py. Writes
comparison_summary.json when the v1 result file already exists.

    conda run -n compo python comparison/run_v2_pretraining_benchmark.py [--parallel N]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.tools import infer_levels
from Compocyte.data import sample_data, sample_hierarchy

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "result", "pretraining")


def peak_rss_bytes():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=0)
    ap.add_argument("--num-threads", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    from Compocyte.core.models.fit_methods import train_resource_budget

    adata = sample_data()
    hierarchy = sample_hierarchy()
    _, obs_names = infer_levels(
        hierarchy=hierarchy, labels='labels', root_node='T_cell', adata=adata)

    runs = []
    for rep in range(args.repeats):
        hc = HierarchicalClassifier(
            save_path=os.path.join(args.outdir, f"v2_model_{rep}"),
            adata=adata.copy(),
            root_node='T_cell',
            dict_of_cell_relations=hierarchy,
            obs_names=obs_names)
        hc.num_threads = args.num_threads
        rss0 = peak_rss_bytes()
        t0 = time.perf_counter()
        if args.parallel > 0:
            hc.train_all_child_nodes(parallelize=True, processes=args.parallel)
        else:
            hc.train_all_child_nodes()
        t1 = time.perf_counter()
        peak = peak_rss_bytes()
        runs.append({
            "repeat": rep,
            "train_s": t1 - t0,
            "rss_before_bytes": rss0,
            "rss_peak_bytes": peak,
            "rss_delta_bytes": max(0, peak - rss0),
        })

    best = min(r["train_s"] for r in runs)
    result = {
        "version": "v2.0-optimized",
        "package_file": os.path.abspath(Compocyte.__file__),
        "git_sha": git_sha("/home/ritwik24222/Compocyte_walle_lab"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": "Compocyte sample TIL data (Beltz et al. 2026), 861 cells x 1011 HVG, 6 subsets",
        "geo_reference": "GSE216280 (human PBMC scRNA-seq, 10x Genomics, GPL20795)",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "hierarchy_nodes": len(hierarchy),
        "epochs": args.epochs,
        "repeats": args.repeats,
        "parallel_processes": args.parallel,
        "num_threads": args.num_threads,
        "resource_budget": train_resource_budget(),
        "best_train_s": best,
        "mean_train_s": float(np.mean([r["train_s"] for r in runs])),
        "runs": runs,
    }
    with open(os.path.join(args.outdir, "v2_pretraining.json"), "w") as f:
        json.dump(result, f, indent=2)

    v1_path = os.path.join(args.outdir, "v1_pretraining.json")
    if os.path.exists(v1_path):
        v1 = json.load(open(v1_path))
        summary = {
            "v1_mean_train_s": v1["mean_train_s"],
            "v2_mean_train_s": result["mean_train_s"],
            "speedup_train": v1["mean_train_s"] / result["mean_train_s"],
            "v1_peak_rss_bytes": max(r["rss_peak_bytes"] for r in v1["runs"]),
            "v2_peak_rss_bytes": max(r["rss_peak_bytes"] for r in runs),
            "v1_rss_delta_bytes": max(r.get("rss_delta_bytes", 0) for r in v1["runs"]),
            "v2_rss_delta_bytes": max(r.get("rss_delta_bytes", 0) for r in runs),
        }
        with open(os.path.join(args.outdir, "comparison_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
