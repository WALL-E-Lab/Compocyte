import hashlib

import numpy as np
import torch
from scipy import sparse

from Compocyte.core.models.fit_methods import (
    SparseBatchDataset,
    _resolve_loader_config,
    _sparse_collate,
    dataloaders_from_dense,
    row_robust_scale,
    row_robust_scale_sparse,
    set_threads,
    train_resource_budget,
)
from Compocyte.data import sample_data


def _xy(n=None):
    base = sample_data()
    X = base.X.tocsr()
    if n is not None and n > X.shape[0]:
        reps = int(np.ceil(n / X.shape[0]))
        X = sparse.vstack([X] * reps, format='csr')[:n]
    n = X.shape[0]
    y = np.zeros((n, 2), dtype=np.float32)
    y[:n // 2, 0] = 1.0
    y[n // 2:, 1] = 1.0
    return X, y


def test_sparse_scaling_matches_dense():
    X, _ = _xy()
    dense = row_robust_scale(X.toarray())
    sparse_scaled = row_robust_scale_sparse(X)
    assert sparse.issparse(sparse_scaled)
    assert np.array_equal(np.asarray(sparse_scaled.todense()), dense)


def test_sparse_dataset_batches_match_dense():
    X, y = _xy(n=2000)
    scaled = row_robust_scale_sparse(X)
    tr_sparse, _ = dataloaders_from_dense(scaled, y, 128, 0, seed=7)
    tr_dense, _ = dataloaders_from_dense(np.asarray(scaled.todense()), y, 128, 0, seed=7)
    bs = list(iter(tr_sparse))
    bd = list(iter(tr_dense))
    assert len(bs) == len(bd)
    # batch composition is drawn from a per-iteration RNG, so compare the
    # multiset of rows over a full epoch rather than per batch
    rows_sparse = sorted(r.numpy().tobytes() for xb, _ in bs for r in xb)
    rows_dense = sorted(r.numpy().tobytes() for xb, _ in bd for r in xb)
    assert rows_sparse == rows_dense
    assert sorted(v.item() for _, yb in bs for v in yb.sum(1)) == sorted(
        v.item() for _, yb in bd for v in yb.sum(1))


def test_sparse_dataset_len_and_dtype():
    X, y = _xy(n=500)
    ds = SparseBatchDataset(X, y)
    assert len(ds) == X.shape[0]
    xb, yb = _sparse_collate([0, 1, 2], ds.X_csr, y)
    assert xb.dtype == torch.float32
    assert xb.shape == (3, X.shape[1])
    assert yb.shape == (3, y.shape[1])


def test_loader_config_respects_memory_budget():
    cfg = _resolve_loader_config(2_000_000, 5000, 512, 4)
    assert cfg['num_workers'] >= 0
    assert cfg['batch_size'] * 5000 * 4 <= max(cfg['ram_budget_bytes'], 1)
    assert cfg['sparse'] in (True, False)
    # Explicitly requested workers are now honored even for tiny matrices,
    # because GPU callers need fork-safe CPU workers to feed the device.
    small = _resolve_loader_config(1000, 50, 64, 4)
    assert small['num_workers'] == 4
    default_zero = _resolve_loader_config(1000, 50, 64, 0)
    assert default_zero['num_workers'] == 0


def test_set_threads_honours_max_threads_cap(monkeypatch):
    monkeypatch.setenv('COMPOCYTE_MAX_THREADS', '4')
    monkeypatch.delenv('COMPOCYTE_NUM_WORKERS', raising=False)
    n = set_threads(32, False)
    assert torch.get_num_threads() <= 4
    assert n >= 0
    monkeypatch.setenv('COMPOCYTE_NUM_WORKERS', '0')
    assert set_threads(32, True) == 0


def test_resource_budget_default_fraction():
    b = train_resource_budget()
    assert b['max_threads'] >= 1
    assert 0 < b['ram_budget_bytes'] < 8 * 1024 ** 4
