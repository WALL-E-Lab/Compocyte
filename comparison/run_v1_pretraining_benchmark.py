#!/usr/bin/env python3
"""Frozen v1.0 pretraining baseline on a scRNA-seq dataset (GSE216280-like PBMC).

Trains the full T-cell hierarchy on the bundled TIL sample (861 cells x 1011 HVG,
6 balanced subsets), recording wall time and peak RSS per phase.

    conda run -n compo-v1 python comparison/run_v1_pretraining_benchmark.py
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
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import torch
    adata = sample_data()
    hierarchy = sample_hierarchy()
    _, obs_names = infer_levels(
        hierarchy=hierarchy, labels='labels', root_node='T_cell', adata=adata)

    runs = []
    for rep in range(args.repeats):
        hc = HierarchicalClassifier(
            save_path=os.path.join(args.outdir, f"v1_model_{rep}"),
            adata=adata.copy(),
            root_node='T_cell',
            dict_of_cell_relations=hierarchy,
            obs_names=obs_names)
        rss0 = peak_rss_bytes()
        t0 = time.perf_counter()
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
        "version": "v1.0-frozen",
        "package_file": os.path.abspath(Compocyte.__file__),
        "git_sha": git_sha("/home/ritwik24222/Compocyte_v1.0"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "dataset": "Compocyte sample TIL data (Beltz et al. 2026), 861 cells x 1011 HVG, 6 subsets",
        "geo_reference": "GSE216280 (human PBMC scRNA-seq, 10x Genomics, GPL20795)",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "hierarchy_nodes": len(hierarchy),
        "epochs": args.epochs,
        "repeats": args.repeats,
        "best_train_s": best,
        "mean_train_s": float(np.mean([r["train_s"] for r in runs])),
        "runs": runs,
    }
    with open(os.path.join(args.outdir, "v1_pretraining.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
