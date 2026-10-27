#!/usr/bin/env python3
"""v2_cpu framework: train a Compocyte hierarchy on a Suco TIL dataset (CPU).

CPU parallelism pipeline: node-level multiprocessing plus the v2.0 training
data-path optimizations (sparse batch collation, decoupled workers, memory
budget). Run with the v2 env only:
    conda run -n compo python comparison/run_train_v2cpu.py --dataset bassez [--parallel 4]

Results land in comparison/result/pretraining/v2_cpu/<dataset>/.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import threading
import time

import numpy as np
import scanpy as sc

import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.tools import infer_levels

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "suco_til")
BASE_OUTDIR = os.path.join(HERE, "result", "pretraining")

DATASETS = {
    "bassez": "TIL-X-BRCA-X-scRNAseq-X-2021-X-Bassez-X-10.1038_s41591-021-01323-8.h5ad",
    "zhang": "TIL-X-BRCA-X-scRNAseq-X-2021-X-Zhang-X-10.1016_j.ccell.2021.09.010.h5ad",
    "che": "TIL-X-COAD-X-scRNAseq-X-2021-X-Che-X-10.1038_s41421-021-00312-y.h5ad",
    "ganesh": "TIL-X-COAD-X-scRNAseq-X-2021-X-Ganesh.h5ad",
    "wang": "TIL-X-COAD-X-scRNAseq-X-2023-X-Wang-X-10.1126_sciadv.adf5464.h5ad",
    "peng": "TIL-X-PAAD-X-scRNAseq-X-2019-X-Peng-X-10.1038_s41422-019-0195-y.h5ad",
    "steele": "TIL-X-PAAD-X-scRNAseq-X-2020-X-Steele-X-10.1038_s43018-020-00121-4.h5ad",
    "werba": "TIL-X-PAAD-X-scRNAseq-X-2023-X-Werba-X-10.1038_s41467-023-36296-4.h5ad",
}


def peak_rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rss_tree_bytes():
    """Sum RSS of this process and its direct children (worker pool)."""
    total = 0
    me = os.getpid()
    page = os.sysconf('SC_PAGE_SIZE')
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            with open(f'/proc/{pid}/stat') as f:
                stat = f.read()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(pid) != me and int(fields[1]) != me:
                continue
            with open(f'/proc/{pid}/statm') as f:
                total += int(f.read().split()[1]) * page
        except (OSError, ValueError, IndexError):
            continue
    return total


class RssTreeSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_bytes = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            self.peak_bytes = max(self.peak_bytes, _rss_tree_bytes())
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)
        return self.peak_bytes


def git_sha(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def stratified_cap(adata, label_col, max_cells, seed):
    rng = np.random.default_rng(seed)
    if max_cells is None or adata.n_obs <= max_cells:
        return adata
    keep = []
    for lab, idx in adata.obs.groupby(label_col).groups.items():
        idx = np.asarray(idx)
        n = max(1, int(round(len(idx) / adata.n_obs * max_cells)))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return adata[np.concatenate(keep)].copy()


def train_test_split_obs(adata, label_col, test_size, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(adata.n_obs)
    try:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                                  stratify=adata.obs[label_col].values)
    except ValueError:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    return adata[tr].copy(), adata[te].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--outdir", default=BASE_OUTDIR)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from Compocyte.core.models.fit_methods import train_resource_budget

    outdir = os.path.join(args.outdir, "v2_cpu", args.dataset)
    os.makedirs(outdir, exist_ok=True)

    adata = sc.read_h5ad(os.path.join(args.data_dir, DATASETS[args.dataset]))
    adata = stratified_cap(adata, "annotation", args.max_cells, args.seed)
    with open(os.path.join(args.data_dir, "hierarchy.pkl"), "rb") as f:
        hierarchy = pickle.load(f)
    train, test = train_test_split_obs(adata, "annotation", args.test_size, args.seed)
    _, obs_names = infer_levels(hierarchy=hierarchy, labels="annotation",
                                root_node="Tumor", adata=train)
    test_true = test.obs[["annotation"]].copy()

    hc = HierarchicalClassifier(
        save_path=os.path.join(outdir, "model"), adata=train,
        root_node="Tumor", dict_of_cell_relations=hierarchy, obs_names=obs_names)
    hc.num_threads = args.num_threads
    fit_kwargs = {"epochs": args.epochs, "batch_size": args.batch_size,
                  "num_threads": args.num_threads}
    nodes = [n for n in hc.graph.nodes
             if len(list(hc.graph.successors(n))) >= 1]
    rss0 = peak_rss()
    sampler = RssTreeSampler()
    sampler.start()
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
    rss_tree_peak = sampler.stop()
    train_peak = peak_rss()

    hc.load_adata(test)
    t2 = time.perf_counter()
    hc.predict_all_child_nodes(hc.root_node)
    t3 = time.perf_counter()

    pred_cols = [c for c in hc.adata.obs.columns if c.endswith("_pred")]
    preds = hc.adata.obs[pred_cols].copy()
    preds["annotation_true"] = test_true["annotation"].values
    preds.to_csv(os.path.join(outdir, "preds.csv.gz"), compression="gzip")

    val_losses = {}
    for node in hc.graph.nodes:
        lc = hc.graph.nodes[node].get("learning_curve")
        if lc is not None and len(lc) > 0:
            val_losses[node] = float(np.min(lc["val_loss"].values))

    total = (t1 - t0) + (t3 - t2)
    timing = {
        "framework": "v2_cpu",
        "dataset": args.dataset,
        "package_file": os.path.abspath(Compocyte.__file__),
        "git_sha": git_sha("/home/ritwik24222/Compocyte_walle_lab"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cpu",
        "parallel_processes": args.parallel,
        "num_threads": args.num_threads,
        "resource_budget": train_resource_budget(),
        "n_train": int(train.n_obs),
        "n_test": int(test.n_obs),
        "n_genes": int(train.n_vars),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_s": t1 - t0,
        "predict_s": t3 - t2,
        "total_s": total,
        "rss_train_peak_bytes": train_peak,
        "rss_train_delta_bytes": max(0, train_peak - rss0),
        "rss_tree_peak_bytes": rss_tree_peak,
        "rss_predict_peak_bytes": peak_rss(),
        "pred_cols": pred_cols,
        "val_losses": val_losses,
    }
    with open(os.path.join(outdir, "timing.json"), "w") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps({k: v for k, v in timing.items()
                      if k not in ("pred_cols", "val_losses")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
