#!/usr/bin/env python3
"""Compare pretraining frameworks for one Suco TIL dataset.

Compares wall time and prediction correctness across v1, v2_cpu and v2_gpu,
using v1 as the reference for correctness.

    python comparison/compare_pretraining.py --dataset bassez
    python comparison/compare_pretraining.py --dataset all
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "result", "pretraining")
DATA_DIR = os.path.join(HERE, "data", "suco_til")
FRAMEWORKS = ["v1", "v2_cpu", "v2_gpu"]
NA = "__NA__"
ACCURACY_TOLERANCE = 0.02


def norm(s):
    return s.replace("", np.nan).fillna(NA).values


def load_fw(base, fw, dataset):
    d = os.path.join(base, fw, dataset)
    timing_path = os.path.join(d, "timing.json")
    preds_path = os.path.join(d, "preds.csv.gz")
    if not (os.path.exists(timing_path) and os.path.exists(preds_path)):
        return None
    timing = json.load(open(timing_path))
    preds = pd.read_csv(preds_path, index_col=0, compression="gzip")
    return {"timing": timing, "preds": preds}


def deepest_col(preds, cols):
    nonempty = {c: int((preds[c].replace("", np.nan).notna()).sum()) for c in cols}
    covered = [c for c in cols if nonempty[c] > 0]
    if not covered:
        return None

    def level_index(col):
        try:
            return int(col.split("_")[1])
        except (IndexError, ValueError):
            return 0

    return max(covered, key=level_index)


def truth_levels(annotation, data_dir):
    import pickle
    from Compocyte.core.tools import infer_levels
    with open(os.path.join(data_dir, "hierarchy.pkl"), "rb") as f:
        hierarchy = pickle.load(f)
    obs, _ = infer_levels(hierarchy=hierarchy, labels=list(annotation),
                          root_node="Tumor")
    return obs


def level_accuracy(preds, truth):
    per_level = {}
    for col in [c for c in preds.columns if c.endswith("_pred")]:
        level = col[: -len("_pred")]
        if level not in truth.columns:
            continue
        p = norm(preds[col])
        t = truth[level].replace("", np.nan).fillna(NA).values
        mask = (p != NA) & (t != NA)
        per_level[col] = {
            "coverage": float((p != NA).mean()),
            "accuracy": float((p[mask] == t[mask]).mean()) if mask.any() else None,
            "n_covered": int((p != NA).sum()),
        }
    return per_level


def compare_one(base, dataset, data_dir=DATA_DIR):
    loaded = {fw: load_fw(base, fw, dataset) for fw in FRAMEWORKS}
    if loaded["v1"] is None:
        return {"dataset": dataset, "error": "v1 results missing"}
    ref = loaded["v1"]
    ref_cols = [c for c in ref["timing"]["pred_cols"]]
    truth = truth_levels(
        ref["preds"]["annotation_true"].fillna("").tolist(), data_dir)

    out = {"dataset": dataset, "frameworks_present": [f for f in FRAMEWORKS
                                                      if loaded[f] is not None]}
    time_table, correctness = {}, {}
    for fw in FRAMEWORKS:
        if loaded[fw] is None:
            continue
        t = loaded[fw]["timing"]
        time_table[fw] = {
            "train_s": t["train_s"], "predict_s": t["predict_s"],
            "total_s": t["total_s"],
            "rss_train_peak_bytes": t.get("rss_train_peak_bytes"),
            "rss_train_delta_bytes": t.get("rss_train_delta_bytes"),
            "rss_tree_peak_bytes": t.get("rss_tree_peak_bytes"),
            "gpu_peak_mb": t.get("gpu_peak_mb"),
            "device_used": t.get("device_used"),
            "parallel_processes": t.get("parallel_processes"),
            "n_train": t["n_train"], "n_test": t["n_test"],
            "epochs": t["epochs"], "batch_size": t.get("batch_size"),
        }
        preds = loaded[fw]["preds"]
        cols = [c for c in t["pred_cols"] if c in ref_cols]
        deep = deepest_col(preds, cols) if cols else None
        correctness[fw] = {
            "deepest_col": deep,
            "per_level_vs_truth": level_accuracy(preds, truth),
        }

    agreement = {}
    for fw in ("v2_cpu", "v2_gpu"):
        if loaded[fw] is None:
            continue
        shared = [c for c in loaded[fw]["timing"]["pred_cols"] if c in ref_cols]
        a = loaded[fw]["preds"]
        b = ref["preds"]
        per_col = {}
        for c in shared:
            per_col[c] = bool((norm(a[c]) == norm(b[c])).all())
        all_match = all(per_col.values()) if per_col else None
        mismatch = None
        if shared and not all_match:
            stacked = np.column_stack([norm(a[c]) for c in shared])
            ref_stacked = np.column_stack([norm(b[c]) for c in shared])
            mismatch = int((stacked != ref_stacked).any(axis=1).sum())
        agreement[fw] = {"shared_cols": shared, "per_col_exact_match": per_col,
                         "all_exact_match": all_match,
                         "n_mismatch_cells": mismatch, "n_test": len(a)}

    parity, config_match = {}, {}
    ref_levels = correctness.get("v1", {}).get("per_level_vs_truth", {})
    for fw in FRAMEWORKS:
        if fw == "v1" or loaded[fw] is None:
            continue
        levels = correctness[fw]["per_level_vs_truth"]
        acc_delta, cov_delta = {}, {}
        for col, vals in levels.items():
            ref_vals = ref_levels.get(col, {})
            if vals.get("accuracy") is not None and ref_vals.get("accuracy") is not None:
                acc_delta[col] = vals["accuracy"] - ref_vals["accuracy"]
            cov_delta[col] = vals["coverage"] - ref_vals.get("coverage", 0.0)
        max_acc = max((abs(v) for v in acc_delta.values()), default=0.0)
        parity[fw] = {
            "per_level_accuracy_delta_vs_v1": acc_delta,
            "per_level_coverage_delta_vs_v1": cov_delta,
            "max_abs_accuracy_delta": max_acc,
            "within_tolerance": bool(max_acc <= ACCURACY_TOLERANCE),
        }
        mismatches = []
        ref_time = time_table["v1"]
        for key in ("n_train", "n_test", "epochs", "batch_size"):
            if time_table[fw].get(key) != ref_time.get(key):
                mismatches.append(f"{key}: {time_table[fw].get(key)} != {ref_time.get(key)}")
        config_match[fw] = {"match": not mismatches, "mismatches": mismatches}

    out["time"] = time_table
    out["correctness_vs_truth"] = {fw: {"deepest_col": correctness[fw]["deepest_col"],
                                        "per_level_vs_truth": correctness[fw]["per_level_vs_truth"]}
                                   for fw in correctness}
    out["training_parity_vs_v1"] = parity
    out["config_match_vs_v1"] = config_match
    out["bit_exactness_vs_v1"] = agreement

    ref_mem = time_table["v1"].get("rss_train_peak_bytes") or 0
    memory = {}
    for fw in time_table:
        if fw == "v1":
            continue
        own = time_table[fw].get("rss_tree_peak_bytes") or time_table[fw].get(
            "rss_train_peak_bytes") or 0
        memory[fw] = {
            "peak_bytes": own,
            "ratio_vs_v1": (own / ref_mem) if ref_mem else None,
            "gpu_peak_mb": time_table[fw].get("gpu_peak_mb"),
        }
    out["memory"] = {"v1_peak_bytes": ref_mem, "frameworks": memory}
    out["notes"] = {
        "accuracy_tolerance": ACCURACY_TOLERANCE,
        "bit_exactness_note": (
            "Bit-exact agreement is only expected when the same frozen weights are "
            "used (inference benchmarks). Independently trained frameworks differ "
            "through RNG, batch order and torch build, so models are compared by "
            "per-level accuracy parity, not label equality."),
    }
    if "v1" in time_table:
        out["speedup_vs_v1"] = {
            fw: {"train": time_table["v1"]["train_s"] / time_table[fw]["train_s"],
                 "total": time_table["v1"]["total_s"] / time_table[fw]["total_s"]}
            for fw in time_table if fw != "v1"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="bassez|zhang|che|ganesh|wang|peng|steele|werba|all")
    ap.add_argument("--basedir", default=BASE)
    args = ap.parse_args()

    if args.dataset == "all":
        datasets = sorted(d for d in os.listdir(os.path.join(args.basedir, "v1"))
                          if os.path.isdir(os.path.join(args.basedir, "v1", d)))
    else:
        datasets = [args.dataset]
    for ds in datasets:
        res = compare_one(args.basedir, ds)
        with open(os.path.join(args.basedir, f"compare_{ds}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"===== {ds} =====")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    sys.exit(main())
