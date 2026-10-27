#!/usr/bin/env python3
"""Compare v2_cpu vs v2_gpu across Suco TIL and PBMC datasets.

Generates per-dataset comparison figures (speedup, per-level accuracy, per-level
coverage) and a combined inference-time bar chart across six datasets.

Outputs:
- comparison/result/v2_cpu_vs_gpu/compare_<dataset>.json
- comparison/figs/<dataset>_v2_cpu_vs_gpu_comparison.{png,pdf}
- comparison/figs/inference_time_v2_cpu_vs_gpu.{png,pdf}

Example:
    conda run -n compo python comparison/compare_v2_cpu_vs_gpu.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE_FIG = HERE / "figs"
BASE_RESULT = HERE / "result" / "v2_cpu_vs_gpu"

DOMAINS = {
    "til": {
        "base": HERE / "result" / "pretraining",
        "data_dir": HERE / "data" / "suco_til",
        "root_node": "Tumor",
    },
    "pbmc": {
        "base": HERE / "result" / "pretraining_pbmc",
        "data_dir": HERE / "data" / "suco_pbmc",
        "root_node": "Blood",
    },
}

FRAMEWORKS = ["v2_cpu", "v2_gpu"]
COLORS = {"v2_cpu": "#4477AA", "v2_gpu": "#228833"}
NA = "__NA__"

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})


def parse_spec(spec):
    """Parse 'name:domain' (domain defaults to 'til')."""
    if ":" in spec:
        name, domain = spec.split(":", 1)
    else:
        name, domain = spec, "til"
    name = name.strip()
    domain = domain.strip()
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}' in '{spec}'. Use: {list(DOMAINS)}")
    return name, domain


def load_fw(domain, fw, dataset):
    base = DOMAINS[domain]["base"]
    d = base / fw / dataset
    timing_path = d / "timing.json"
    preds_path = d / "preds.csv.gz"
    if not (timing_path.exists() and preds_path.exists()):
        return None
    timing = json.load(open(timing_path))
    preds = pd.read_csv(preds_path, index_col=0, compression="gzip")
    return {"timing": timing, "preds": preds}


def truth_levels(annotation, domain):
    import pickle
    from Compocyte.core.tools import infer_levels
    data_dir = DOMAINS[domain]["data_dir"]
    root_node = DOMAINS[domain]["root_node"]
    with open(data_dir / "hierarchy.pkl", "rb") as f:
        hierarchy = pickle.load(f)
    obs, _ = infer_levels(hierarchy=hierarchy, labels=list(annotation),
                          root_node=root_node)
    return obs


def level_label(col):
    level = col[: -len("_pred")]
    m = re.match(r"Level_(\d+)", level)
    return f"L{m.group(1)}" if m else level


def norm(s):
    return s.replace("", np.nan).fillna(NA).values


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


def compare_one(dataset, domain):
    loaded = {fw: load_fw(domain, fw, dataset) for fw in FRAMEWORKS}
    missing = [fw for fw, v in loaded.items() if v is None]
    if missing:
        print(f"Skipping {dataset}: missing {missing}")
        return None
    ref = loaded["v2_cpu"]
    truth = truth_levels(ref["preds"]["annotation_true"].fillna("").tolist(),
                         domain)

    out = {"dataset": dataset, "domain": domain}
    time_table = {}
    correctness = {}
    for fw in FRAMEWORKS:
        t = loaded[fw]["timing"]
        time_table[fw] = {
            "train_s": t["train_s"],
            "predict_s": t["predict_s"],
            "total_s": t["total_s"],
            "rss_train_peak_bytes": t.get("rss_train_peak_bytes"),
            "rss_tree_peak_bytes": t.get("rss_tree_peak_bytes"),
            "gpu_peak_mb": t.get("gpu_peak_mb"),
            "device_used": t.get("device_used"),
            "parallel_processes": t.get("parallel_processes"),
            "n_train": t["n_train"],
            "n_test": t["n_test"],
            "epochs": t["epochs"],
            "batch_size": t.get("batch_size"),
        }
        preds = loaded[fw]["preds"]
        correctness[fw] = {
            "per_level_vs_truth": level_accuracy(preds, truth),
        }

    out["time"] = time_table
    out["correctness_vs_truth"] = correctness
    out["speedup_gpu_vs_cpu"] = {
        "train": time_table["v2_cpu"]["train_s"] / time_table["v2_gpu"]["train_s"],
        "total": time_table["v2_cpu"]["total_s"] / time_table["v2_gpu"]["total_s"],
        "predict": time_table["v2_cpu"]["predict_s"] / time_table["v2_gpu"]["predict_s"],
    }
    return out


def plot_dataset_figure(data, dataset, out_dir):
    speedups = [data["speedup_gpu_vs_cpu"]["train"],
                data["speedup_gpu_vs_cpu"]["total"]]

    levels = sorted(
        data["correctness_vs_truth"]["v2_cpu"]["per_level_vs_truth"].keys(),
        key=lambda c: int(re.search(r"\d+", c).group()) if re.search(r"\d+", c) else 0
    )
    accuracy = {fw: [] for fw in FRAMEWORKS}
    coverage = {fw: [] for fw in FRAMEWORKS}
    for fw in FRAMEWORKS:
        for lvl in levels:
            vals = data["correctness_vs_truth"][fw]["per_level_vs_truth"][lvl]
            accuracy[fw].append(vals["accuracy"] * 100 if vals["accuracy"] is not None else 0.0)
            coverage[fw].append(vals["coverage"] * 100)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
                          hspace=0.4, wspace=0.3)

    # Speedup panel
    ax_spd = fig.add_subplot(gs[:, 0])
    x = np.arange(2)
    width = 0.35
    bars = ax_spd.bar(x, speedups, width, color=COLORS["v2_gpu"],
                      edgecolor="black", linewidth=0.5)
    ax_spd.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_spd.set_xlabel("Phase", fontsize=10)
    ax_spd.set_ylabel("Speedup (\u00d7v2_cpu)", fontsize=10)
    ax_spd.set_xticks(x)
    ax_spd.set_xticklabels(["Training", "Total"])
    ax_spd.set_ylim(0, max(speedups + [1.0]) * 1.2)
    ax_spd.set_title(f"Dataset: {dataset.capitalize()}", fontsize=11,
                     fontweight="bold", pad=10)
    for bar, val in zip(bars, speedups):
        ax_spd.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.05,
                    f"{val:.1f}\u00d7", ha="center", va="bottom", fontsize=9)

    # Accuracy panel
    ax_acc = fig.add_subplot(gs[0, 1])
    x = np.arange(len(levels))
    width = 0.35
    ax_acc.bar(x - width / 2, accuracy["v2_cpu"], width,
               color=COLORS["v2_cpu"], edgecolor="black", linewidth=0.5,
               label="v2.0 (CPU)")
    ax_acc.bar(x + width / 2, accuracy["v2_gpu"], width,
               color=COLORS["v2_gpu"], edgecolor="black", linewidth=0.5,
               label="v2.0 (GPU)")
    ax_acc.set_ylabel("Accuracy (%)", fontsize=10)
    ax_acc.set_ylim(0, 110)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([level_label(l) for l in levels], rotation=0)
    ax_acc.legend(loc="lower left", frameon=False, fontsize=8)
    ax_acc.set_title("Per-Level Accuracy", fontsize=10, pad=5)

    # Coverage panel
    ax_cov = fig.add_subplot(gs[1, 1])
    ax_cov.bar(x - width / 2, coverage["v2_cpu"], width,
               color=COLORS["v2_cpu"], edgecolor="black", linewidth=0.5)
    ax_cov.bar(x + width / 2, coverage["v2_gpu"], width,
               color=COLORS["v2_gpu"], edgecolor="black", linewidth=0.5)
    ax_cov.set_ylabel("Coverage (%)", fontsize=10)
    ax_cov.set_xlabel("Hierarchy Level", fontsize=10)
    ax_cov.set_ylim(0, 110)
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels([level_label(l) for l in levels], rotation=0)
    ax_cov.set_title("Per-Level Coverage", fontsize=10, pad=5)

    n_train = data["time"]["v2_cpu"]["n_train"]
    n_test = data["time"]["v2_cpu"]["n_test"]
    fig.text(0.02, 0.02, f"n_train={n_train:,}  n_test={n_test:,}",
             fontsize=7, style="italic", color="gray")

    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{dataset}_v2_cpu_vs_gpu_comparison.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{dataset}_v2_cpu_vs_gpu_comparison.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {dataset} comparison figure")


def plot_inference_figure(all_data, dataset_specs, out_dir):
    datasets = [spec[0] for spec in dataset_specs]

    cpu_times, gpu_times, n_tests = [], [], []
    for ds in datasets:
        data = all_data.get(ds)
        if data is None:
            cpu_times.append(None)
            gpu_times.append(None)
            n_tests.append(None)
            continue
        cpu_times.append(data["time"]["v2_cpu"]["predict_s"])
        gpu_times.append(data["time"]["v2_gpu"]["predict_s"])
        n_tests.append(data["time"]["v2_cpu"]["n_test"])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.35

    cpu_vals = [t if t is not None else 0 for t in cpu_times]
    gpu_vals = [t if t is not None else 0 for t in gpu_times]

    b1 = ax.bar(x - width / 2, cpu_vals, width, color=COLORS["v2_cpu"],
                edgecolor="black", linewidth=0.5, label="v2.0 (CPU)")
    b2 = ax.bar(x + width / 2, gpu_vals, width, color=COLORS["v2_gpu"],
                edgecolor="black", linewidth=0.5, label="v2.0 (GPU)")

    ax.set_yscale("log")
    ax.set_ylabel("Inference wall time (s, log scale)", fontsize=11)
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{d.capitalize()}\n(n={n_tests[i]:,})" if n_tests[i] is not None
        else d.capitalize()
        for i, d in enumerate(datasets)
    ])
    ax.set_title("Inference time: v2.0 CPU vs GPU (test split)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    for bars, vals, base in [(b1, cpu_vals, None), (b2, gpu_vals, cpu_vals)]:
        for bar, val, ref in zip(bars, vals, base if base else vals):
            if val <= 0:
                continue
            if base is None:
                label = f"{val:.2f} s"
            else:
                ratio = ref / val if val > 0 else 0
                label = f"{val:.2f} s\n{ratio:.1f}\u00d7"
            ax.text(bar.get_x() + bar.get_width() / 2., val * 1.12, label,
                    ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_ylim(top=max([v for v in cpu_vals + gpu_vals if v > 0], default=1) * 8)
    fig.tight_layout()
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "inference_time_v2_cpu_vs_gpu.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "inference_time_v2_cpu_vs_gpu.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved inference-time figure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["wang:til", "oelen_e1:pbmc", "terekhova:pbmc"],
                    help="Per-dataset comparison specs: name:domain")
    ap.add_argument("--inference-datasets", nargs="+",
                    default=["steele:til", "che:til", "wang:til", "bassez:til",
                             "oelen_e1:pbmc", "terekhova:pbmc"],
                    help="Datasets for inference-time chart: name:domain")
    args = ap.parse_args()

    BASE_RESULT.mkdir(parents=True, exist_ok=True)
    BASE_FIG.mkdir(exist_ok=True)

    dataset_specs = [parse_spec(s) for s in args.datasets]
    inference_specs = [parse_spec(s) for s in args.inference_datasets]

    print("Comparing v2_cpu vs v2_gpu...\n")
    all_data = {}
    for dataset, domain in dataset_specs:
        data = compare_one(dataset, domain)
        if data is None:
            continue
        all_data[dataset] = data
        with open(BASE_RESULT / f"compare_{dataset}.json", "w") as f:
            json.dump(data, f, indent=2)

    if not all_data:
        print("No dataset had both v2_cpu and v2_gpu results. Exiting.")
        return

    print("\nGenerating figures...")
    for dataset, _ in dataset_specs:
        if dataset in all_data:
            plot_dataset_figure(all_data[dataset], dataset, BASE_FIG)

    # Load any additional inference datasets not in the per-dataset comparison
    for dataset, domain in inference_specs:
        if dataset not in all_data:
            data = compare_one(dataset, domain)
            if data is not None:
                all_data[dataset] = data

    plot_inference_figure(all_data, inference_specs, BASE_FIG)
    print(f"\nAll outputs in {BASE_RESULT} and {BASE_FIG}")


if __name__ == "__main__":
    sys.exit(main())
