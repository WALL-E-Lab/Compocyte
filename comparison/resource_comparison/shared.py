#!/usr/bin/env python3
"""Shared utilities for the resource-comparison benchmark.

All method scripts use these helpers so that every method trains and predicts
on the exact same cells:
  - load_subset() reads a source Suco PBMC h5ad and deterministically subsets
    it to the requested cell count (seed=42, stratified by annotation).
  - export_for_r() writes a temporary, HVG-filtered, gzip-compressed sparse MTX
    plus gene/cell/label sidecars for the R-based methods (CHETAH, transferData).
  - PeakMemorySampler gives a phase-scoped peak resident memory measurement.
  - get_hardware_info() records node/GPU/RAM/CPU so the comparison is reproducible.
"""
import gzip
import json
import os
import platform
import resource
import subprocess
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmwrite
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(os.path.dirname(HERE), "data", "suco_pbmc_extracted")
RESULTS_DIR = os.path.join(HERE, "results")

# Subset key -> (source filename, target number of cells)
SUBSET_CONFIG = {
    "liu_100k": (
        "PBMC-X-COVID-X-scRNAseq-X-2021-X-Liu-X-10.1016_j.cell.2021.02.018.h5ad",
        100_000,
    ),
    "zhang_250k": (
        "PBMC-X-BRCA-X-scRNAseq-X-2021-X-Zhang-X-10.1016_j.ccell.2021.09.010.h5ad",
        250_000,
    ),
    "oelen_e1_500k": (
        "PBMC-X-Infectious-X-scRNAseq-X-2022-X-OELEN_E1-X-10.1038_s41467-022-30893-5.h5ad",
        500_000,
    ),
}


# -----------------------------------------------------------------------------
# Memory sampling
# -----------------------------------------------------------------------------

class PeakMemorySampler(threading.Thread):
    """Sample peak RSS of the whole process tree in a background thread.

    Optionally also samples the *device-level* GPU memory in use via
    nvidia-smi.  This is required for GPU methods because the training workers
    run in separate processes: ``torch.cuda.max_memory_allocated()`` only
    reports the memory allocated by the calling process, so it cannot see the
    VRAM used by forked workers.

    Usage:
        sampler = PeakMemorySampler(gpu=True)
        sampler.start()
        # ... work ...
        peak_bytes = sampler.stop()
    """

    def __init__(self, interval=0.2, gpu=False, gpu_interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.gpu = gpu
        self.gpu_interval = gpu_interval
        self.peak_bytes = 0
        self.gpu_peak_bytes = 0
        self._stop_event = threading.Event()

    def run(self):
        me = os.getpid()
        page = os.sysconf("SC_PAGE_SIZE")
        next_gpu = 0.0
        while not self._stop_event.is_set():
            total = 0
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/stat") as f:
                        stat = f.read()
                    fields = stat[stat.rfind(")") + 2 :].split()
                    # fields[1] is ppid; include this process and its children
                    if int(pid) != me and int(fields[1]) != me:
                        continue
                    with open(f"/proc/{pid}/statm") as f:
                        total += int(f.read().split()[1]) * page
                except (OSError, ValueError, IndexError):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)

            if self.gpu:
                now = time.monotonic()
                if now >= next_gpu:
                    self.gpu_peak_bytes = max(self.gpu_peak_bytes, _gpu_used_bytes())
                    next_gpu = now + self.gpu_interval

            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)
        return self.peak_bytes


def _gpu_used_bytes():
    """Total GPU device memory currently in use (bytes), across all processes."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        total = 0
        for line in out.strip().splitlines():
            line = line.strip()
            if line:
                total += int(line) * 1024 * 1024
        return total
    except Exception:
        return 0


def peak_rss_bytes():
    """Legacy helper: whole-process peak RSS via getrusage."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# -----------------------------------------------------------------------------
# Hardware metadata
# -----------------------------------------------------------------------------

def _read_cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _read_ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except Exception:
        pass
    return None


def _read_gpu_info():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "model": parts[0],
            "driver": parts[1],
            "memory_total": parts[2],
            "memory_free": parts[3],
        }
    except Exception:
        return None


def get_hardware_info():
    """Return a reproducible hardware description for the current node."""
    return {
        "hostname": platform.node(),
        "cpu_model": _read_cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _read_ram_gb(),
        "gpu": _read_gpu_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# -----------------------------------------------------------------------------
# Data loading / export
# -----------------------------------------------------------------------------

def _stratified_subset(adata, label_col, n_cells, seed):
    rng = np.random.default_rng(seed)
    if adata.n_obs <= n_cells:
        return adata.copy()
    keep = []
    groups = adata.obs.groupby(label_col, observed=True).indices
    total = adata.n_obs
    for lab, idx in groups.items():
        idx = np.asarray(idx)
        n = max(1, int(round(len(idx) / total * n_cells)))
        n = min(n, len(idx))
        keep.append(rng.choice(idx, size=n, replace=False))
    keep = np.concatenate(keep)
    if len(keep) > n_cells:
        keep = rng.choice(keep, size=n_cells, replace=False)
    elif len(keep) < n_cells:
        extra = n_cells - len(keep)
        pool = np.setdiff1d(np.arange(adata.n_obs), keep)
        keep = np.concatenate([keep, rng.choice(pool, size=extra, replace=False)])
    return adata[keep].copy()


def _ensure_dataset_col(adata, dataset_name):
    if "dataset" not in adata.obs.columns:
        if "sample_id" in adata.obs.columns:
            adata.obs["dataset"] = adata.obs["sample_id"].astype(str)
        else:
            adata.obs["dataset"] = dataset_name
    return adata


def load_subset(subset_name, seed=42, test_size=0.2):
    """Load source h5ad, deterministically subset, split train/test."""
    if subset_name not in SUBSET_CONFIG:
        raise ValueError(f"Unknown subset {subset_name}. Choose from {list(SUBSET_CONFIG)}")
    filename, n_cells = SUBSET_CONFIG[subset_name]
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    adata = sc.read_h5ad(path)
    adata = _ensure_dataset_col(adata, subset_name)
    adata = _stratified_subset(adata, "annotation", n_cells, seed)

    idx = np.arange(adata.n_obs)
    try:
        tr, te = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=adata.obs["annotation"].values
        )
    except ValueError:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)

    return adata[tr].copy(), adata[te].copy()


def export_for_r(train, test, out_dir, n_top_genes=5000):
    """Export HVG-filtered train/test data as gzip-compressed sparse MTX for R.

    Files written:
        {train,test}_data.mtx.gz  (genes x cells)
        {train,test}_genes.txt
        {train,test}_cells.txt
        {train,test}_labels.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    tmp = train.copy()
    sc.pp.highly_variable_genes(tmp, inplace=True, flavor="seurat_v3", n_top_genes=n_top_genes)
    hvg = tmp.var_names[tmp.var["highly_variable"]].tolist()

    for prefix, adata in [("train", train), ("test", test)]:
        adata_hvg = adata[:, hvg].copy()
        X = adata_hvg.X
        if hasattr(X, "tocsc"):
            X = X.tocsc()
        else:
            X = sp.csc_matrix(X)
        # MTX orientation: rows = genes, cols = cells
        mtx_path = os.path.join(out_dir, f"{prefix}_data.mtx.gz")
        with gzip.open(mtx_path, "wb") as f:
            mmwrite(f, X.T)
        with open(os.path.join(out_dir, f"{prefix}_genes.txt"), "w") as f:
            f.write("\n".join(adata_hvg.var_names.astype(str)) + "\n")
        with open(os.path.join(out_dir, f"{prefix}_cells.txt"), "w") as f:
            f.write("\n".join(adata_hvg.obs_names.astype(str)) + "\n")
        labels = pd.DataFrame({"label": adata_hvg.obs["annotation"].values})
        labels.to_csv(os.path.join(out_dir, f"{prefix}_labels.csv"), index=False)
        meta = pd.DataFrame({"sample_id": adata_hvg.obs["sample_id"].astype(str).values})
        meta.to_csv(os.path.join(out_dir, f"{prefix}_metadata.csv"), index=False)


def write_json(result, out_dir=RESULTS_DIR):
    """Write a result dict as a JSON file; append hardware metadata if absent."""
    os.makedirs(out_dir, exist_ok=True)
    if "hardware" not in result:
        result["hardware"] = get_hardware_info()
    subset = result["subset"]
    method = result["method"].replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(out_dir, f"{subset}_{method}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def make_r_temp_dir():
    return tempfile.mkdtemp(prefix="rc_")


def cleanup_r_temp_dir(path):
    import shutil

    shutil.rmtree(path, ignore_errors=True)
