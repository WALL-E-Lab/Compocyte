#!/usr/bin/env python3
"""Benchmark GPU treeArches (scArches + scHPL) training/inference time and memory.

Usage:
    python train_treearches_gpu.py --subset oelen_e1_500k

The SCVI reference and query training are forced onto the first CUDA device
via `accelerator="gpu"` / `devices=1`.  The scHPL hierarchical classifier
itself is CPU-based; GPU memory therefore mainly reflects the SCVI steps.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import time

# treeArches runs in the `scrna` env where Compocyte is not installed.
# Add the repository src/ tree to PYTHONPATH so we can import only the helpers
# we need (dict_depth, infer_levels, make_graph_from_edges).
# The `scrna` env additionally needs `pooch` installed:
#     conda run -n scrna pip install pooch
import importlib.util as _util

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Compocyte's pretrained module imports pooch at package import time.  If pooch
# is missing, provide a minimal stub so the rest of the package (which is all
# we use here) can still be imported.
if _util.find_spec("pooch") is None:
    import types

    _pooch_stub = types.ModuleType("pooch")

    def _retrieve(*args, **kwargs):
        raise RuntimeError("pooch is not installed; run: conda run -n scrna pip install pooch")

    _pooch_stub.retrieve = _retrieve
    _pooch_stub.fetch = _retrieve
    sys.modules["pooch"] = _pooch_stub

import networkx as nx
import numpy as np
import scanpy as sc
import torch
from scarches.classifiers import scHPL
from scarches.models import SCVI

import Compocyte
from Compocyte.core.tools import dict_depth, infer_levels, make_graph_from_edges

from shared import PeakMemorySampler, load_subset, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results", "gpu")
ROOT_NODE = "Blood"

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
        raise RuntimeError("CUDA is not available in this environment; cannot run treeArches GPU benchmark.")

    os.makedirs(args.out_dir, exist_ok=True)
    train, test = load_subset(args.subset, seed=args.seed, test_size=args.test_size)

    hierarchy_path = os.path.join(
        os.path.dirname(HERE), "data", "suco_pbmc_extracted", "hierarchy.pkl"
    )
    with open(hierarchy_path, "rb") as f:
        hierarchy = pickle.load(f)
    graph = nx.DiGraph()
    make_graph_from_edges(hierarchy, graph)

    train.obs["last_label"] = train.obs["annotation"].astype(str)
    train.obs["batch"] = train.obs["dataset"].astype(str)
    test.obs["last_label"] = test.obs["annotation"].astype(str)
    test.obs["batch"] = test.obs["dataset"].astype(str)

    combined = sc.concat([train, test], label="split", keys=["train", "test"])
    combined.obs["last_label"] = combined.obs["last_label"].astype(str)
    combined.obs["batch"] = combined.obs["batch"].astype(str)

    obs_levels, _ = infer_levels(
        hierarchy=hierarchy, labels="annotation", root_node=ROOT_NODE, adata=combined
    )
    for col in obs_levels.columns:
        combined.obs[col] = obs_levels[col].values

    sc.pp.highly_variable_genes(
        combined, n_top_genes=2000, inplace=True, batch_key=None, subset=True, flavor="seurat_v3"
    )

    labels = combined.obs["last_label"].values
    labels_unique, counts = np.unique(labels, return_counts=True)
    singleton_labels = labels_unique[counts <= 5]
    combined = combined[~combined.obs["last_label"].isin(singleton_labels)].copy()

    # Only keep the hierarchy nodes that lie on a path from the root to a cell
    # type that is actually present in the data.  The full Suco hierarchy has
    # 86 nodes but only ~47 are reachable from the cell types present here
    # (branches such as ERY, PL, neutro, baso, ... contain no cells).  scHPL's
    # tree training crashes on such empty branches with
    # "ValueError: zero-size array to reduction operation minimum".
    observed_labels = combined.obs["last_label"].unique().tolist()
    keep_nodes = {ROOT_NODE}
    for label in observed_labels:
        if label not in graph:
            raise ValueError(f"Cell type '{label}' is not part of the hierarchy.")
        keep_nodes.update(nx.shortest_path(graph, ROOT_NODE, label))

    train_mask = combined.obs["split"] == "train"
    train_adata = combined[train_mask].copy()
    test_adata = combined[~train_mask].copy()

    SCVI.setup_anndata(combined, batch_key="batch")
    vae = SCVI(
        combined,
        n_layers=2,
        encode_covariates=True,
        deeply_inject_covariates=True,
        use_layer_norm="both",
        use_batch_norm="none",
    )

    # Training phase
    train_sampler = PeakMemorySampler(gpu=True)
    train_sampler.start()
    t0 = time.perf_counter()
    vae.train(max_epochs=args.epochs, **GPU_TRAIN_KWARGS)
    reference_latent = sc.AnnData(vae.get_latent_representation())
    obs_names = [f"Level_{i}" for i in range(dict_depth(hierarchy))]
    for obs_name in obs_names:
        reference_latent.obs[obs_name] = combined.obs[obs_name].tolist()
    reference_latent.obs["batch"] = combined.obs["batch"].tolist()
    reference_latent.obs["last_label"] = combined.obs["last_label"].tolist()

    # Build the scHPL tree from the (pruned) hierarchy.
    pbmc_tree = scHPL.utils.create_tree("root")
    node_map = {ROOT_NODE: pbmc_tree[0]}

    def add_schpl_children(parent_name, parent_node):
        for child_name in graph.successors(parent_name):
            if child_name not in keep_nodes:
                continue
            if child_name not in node_map:
                child_node = scHPL.utils.TreeNode([child_name])
                parent_node.add_descendant(child_node)
                node_map[child_name] = child_node
            add_schpl_children(child_name, node_map[child_name])

    add_schpl_children(ROOT_NODE, pbmc_tree[0])
    pbmc_tree_t = scHPL.train.train_tree(
        reference_latent.X,
        np.array(reference_latent.obs["last_label"].values),
        pbmc_tree,
        classifier="knn",
        dimred=False,
        useRE=True,
        FN=0.5,
        n_neighbors=50,
        dynamic_neighbors=True,
    )
    t1 = time.perf_counter()
    train_peak = train_sampler.stop()
    gpu_train_peak = train_sampler.gpu_peak_bytes

    # Inference phase
    inf_sampler = PeakMemorySampler(gpu=True)
    inf_sampler.start()
    t2 = time.perf_counter()
    model = SCVI.load_query_data(test_adata, vae, freeze_dropout=True)
    model.train(max_epochs=args.epochs, **GPU_TRAIN_KWARGS)
    query_latent = sc.AnnData(model.get_latent_representation())
    query_pred = scHPL.predict_labels(query_latent.X, tree=pbmc_tree_t)
    t3 = time.perf_counter()
    inference_peak = inf_sampler.stop()
    gpu_inf_peak = inf_sampler.gpu_peak_bytes

    result = {
        "method": "treeArches",
        "subset": args.subset,
        "n_train": int(train_adata.n_obs),
        "n_test": int(test_adata.n_obs),
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
