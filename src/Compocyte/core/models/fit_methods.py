from copy import deepcopy
import os
import time
from typing import Union
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import (
        TensorDataset, random_split, DataLoader, IterableDataset,
        Dataset, get_worker_info,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    _TORCH_AVAILABLE = False

    class _TorchPlaceholder:
        """Stands in for torch classes when torch is not installed.

        Importing this module must succeed so that sklearn-only paths keep
        working; only touching torch functionality raises, with a pointer
        to the right extra.
        """
        def __init__(self, *args, **kwargs):
            _require_torch()

    TensorDataset = random_split = DataLoader = get_worker_info = _missing_torch_callable = None  # placeholders, replaced below
    IterableDataset = Dataset = _TorchPlaceholder


def _missing_torch_callable(*args, **kwargs):
    _require_torch()


TensorDataset = TensorDataset if _TORCH_AVAILABLE else _missing_torch_callable
random_split = random_split if _TORCH_AVAILABLE else _missing_torch_callable
DataLoader = DataLoader if _TORCH_AVAILABLE else _missing_torch_callable
get_worker_info = get_worker_info if _TORCH_AVAILABLE else _missing_torch_callable


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "This code path requires the 'torch' extra: "
            "pip install \"Compocyte[torch]\"")
    return torch


def _require_dask():
    try:
        import dask.array as da
    except ImportError as e:
        raise ImportError(
            "The out-of-core (dask) training path requires the 'atlas' extra: "
            "pip install \"Compocyte[atlas]\"") from e
    return da


def _require_balanced_loss():
    try:
        from balanced_loss import Loss as BalancedLoss
    except ImportError as e:
        raise ImportError(
            "Class-balanced focal loss requires the 'focal' extra: "
            "pip install \"Compocyte[focal]\"") from e
    return BalancedLoss
import logging
from functools import partial
from Compocyte.core.models.dense_torch import DenseTorch, resolve_device
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.models.trees import BoostedTrees
from scipy import stats

logger = logging.getLogger(__name__)

# Serializes model invocations so that multithreaded callers (e.g. level-
# parallel prediction) observe the same BLAS/thread-pool state as sequential
# execution, keeping predictions bit-identical. Uncontended overhead is ~ns.
import threading as _threading
_FORWARD_LOCK = _threading.Lock()

# Normal IQR adjustment used by sklearn's RobustScaler with the default
# quantile_range=(25.0, 75.0) and unit_variance=True.
_IQR_ADJUST = float(stats.norm.ppf(0.75) - stats.norm.ppf(0.25))
_USE_FAST_SCALE = os.environ.get('COMPOCYTE_FAST_SCALE', '1') == '1'
_USE_SPARSE_TRAIN = os.environ.get('COMPOCYTE_SPARSE_TRAIN', '1') == '1'


def _total_ram_bytes():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except Exception:
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
    return 8 * 1024 ** 3


def train_resource_budget():
    max_threads = int(os.environ.get('COMPOCYTE_MAX_THREADS', '16'))
    ram_frac = float(os.environ.get('COMPOCYTE_MAX_RAM_FRACTION', '0.5'))
    ram_frac = min(max(ram_frac, 0.05), 0.95)
    return {
        'max_threads': max(1, max_threads),
        'ram_budget_bytes': int(_total_ram_bytes() * ram_frac),
    }


def peak_rss_bytes():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception:
        return 0


def current_rss_bytes():
    try:
        with open('/proc/self/statm') as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf('SC_PAGE_SIZE')
    except Exception:
        return 0


def _resolve_loader_config(n_cells, n_genes, batch_size, num_workers, device=None):
    budget = train_resource_budget()
    dense_bytes = n_cells * n_genes * 4
    batch_bytes = batch_size * n_genes * 4
    workers = int(num_workers)
    # Trust the caller's worker choice. GPU callers pass num_workers>0 so that
    # fork-safe CPU workers prefetch and pin batches while the main process
    # owns CUDA; CPU callers that run nodes in a process pool pass 0 workers.
    prefetch = 2
    if dense_bytes > budget['ram_budget_bytes']:
        workers = min(workers, 2)
    while workers > 0 and batch_bytes * workers * prefetch > budget['ram_budget_bytes']:
        workers -= 1
    while batch_bytes > budget['ram_budget_bytes'] and batch_size > 1:
        batch_size //= 2
        batch_bytes = batch_size * n_genes * 4
    return {
        'batch_size': max(1, int(batch_size)),
        'num_workers': max(0, workers),
        'prefetch_factor': prefetch,
        'sparse': dense_bytes > budget['ram_budget_bytes'],
        'dense_bytes': dense_bytes,
        'ram_budget_bytes': budget['ram_budget_bytes'],
    }


def _to_dense(X):
    """Densify sparse matrices, pass dense arrays through (as ndarray)."""
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def row_robust_scale(X):
    """Cell-wise IQR scaling, bit-identical to
    ``robust_scale(X, axis=1, with_centering=False, copy=False,
    unit_variance=True)``.

    sklearn computes row quantiles with a Python loop over columns of the
    transposed sparse matrix (one ``nanpercentile`` call per cell). This
    instead densifies once and issues a single vectorized ``percentile``
    call, which runs the same per-slice interpolation kernel and therefore
    yields bitwise-identical scales and outputs. Matrices containing NaNs
    fall back to sklearn. Set ``COMPOCYTE_FAST_SCALE=0`` to always use
    sklearn.
    """
    if not _USE_FAST_SCALE:
        return robust_scale(
            X, axis=1, with_centering=False, copy=False, unit_variance=True)

    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.array(X, copy=True)
    if X.ndim != 2 or X.shape[0] == 0:
        return X
    if X.dtype.kind not in 'f':
        X = X.astype(np.float64)
    if np.isnan(X).any():
        return np.asarray(robust_scale(
            X, axis=1, with_centering=False, copy=False, unit_variance=True))

    q = np.percentile(X, (25.0, 75.0), axis=1)
    scale = q[1] - q[0]
    scale[scale < 10 * np.finfo(scale.dtype).eps] = 1.0
    scale = scale / _IQR_ADJUST
    X /= scale[:, None]
    return X


def row_robust_scale_sparse(X, chunk_rows=4096):
    """Row-wise IQR scaling that keeps the result sparse.

    Row scaling divides every entry by a per-row scalar, so zero entries stay
    zero and the CSR structure is preserved. Scaling is applied per row chunk
    through :func:`row_robust_scale`, so the numerics are identical to the
    dense path while peak dense memory is bounded by ``chunk_rows`` rows
    instead of the whole matrix.
    """
    if not _USE_SPARSE_TRAIN:
        return row_robust_scale(X)

    X = X.tocsr()
    if X.shape[0] == 0:
        return X
    chunk_rows = max(1, int(chunk_rows))
    blocks = []
    for start in range(0, X.shape[0], chunk_rows):
        block = row_robust_scale(X[start:start + chunk_rows].toarray())
        blocks.append(sparse.csr_matrix(block))
    return sparse.vstack(blocks, format='csr') if len(blocks) > 1 else blocks[0]


def row_robust_scale_sparse_to_scale(X, chunk_rows=4096):
    """Return only the per-row IQR scale factors (no scaled matrix copy).

    Used by training paths that apply scaling inside the batch collate so that
    workers do not each materialize a full scaled copy of the node matrix.
    Numerics are identical to :func:`row_robust_scale_sparse`; zeros stay
    zero, so multiplying a batch by the scale vector in collate is equivalent.
    """
    if not _USE_SPARSE_TRAIN:
        # dense path: compute via row_robust_scale and extract scale
        scaled = row_robust_scale(X)
        # row_robust_scale modified X in place or returned a copy; reconstruct
        # scale by comparing one non-trivial row. Simpler: compute scale directly.
        X = np.asarray(X)
        q = np.percentile(X, (25.0, 75.0), axis=1)
        scale = q[1] - q[0]
        scale[scale < 10 * np.finfo(scale.dtype).eps] = 1.0
        scale = scale / _IQR_ADJUST
        return scale

    X = X.tocsr()
    n = X.shape[0]
    if n == 0:
        return np.ones(0)
    scales = np.empty(n, dtype=np.float64)
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        block = X[start:end].toarray()
        q = np.percentile(block, (25.0, 75.0), axis=1)
        scale = q[1] - q[0]
        scale[scale < 10 * np.finfo(scale.dtype).eps] = 1.0
        scale = scale / _IQR_ADJUST
        scales[start:end] = scale
    return scales

def to_categorical(y, num_classes, dtype="float32"):
    """
    Simplified from keras to avoid dependency and premature conversion to a Tensor.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    y = y.reshape(-1)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


class DaskBatchDataset(IterableDataset):
    def __init__(self, X, y):
        # Convert both to lists of delayed chunks
        self.X_chunks = X.to_delayed().ravel()
        self.y_chunks = y.to_delayed().ravel()
        self._epoch = 0

        assert len(self.X_chunks) == len(self.y_chunks), \
            "Feature and label chunks must be aligned"

    def set_epoch(self, epoch):
        # Call before each epoch so __iter__ uses a fresh permutation.
        # Must be called in the main process before the DataLoader starts iterating
        # (before workers are spawned), so the updated value is pickled into each worker.
        self._epoch = epoch

    def __iter__(self):
        # All workers derive the same permutation from the epoch seed, then each takes
        # a disjoint slice — this gives epoch-level shuffling that is safe with num_workers > 0.
        rng = np.random.default_rng(self._epoch)
        perm = rng.permutation(len(self.X_chunks))
        X_shuffled = self.X_chunks[perm]
        y_shuffled = self.y_chunks[perm]

        worker_info = get_worker_info()
        if worker_info is None:
            chunk_iter = zip(X_shuffled, y_shuffled)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            chunk_iter = zip(
                X_shuffled[worker_id::num_workers],
                y_shuffled[worker_id::num_workers]
            )

        window_size = 5
        buf_X, buf_y, chunk_sizes = [], [], []
        pending = []

        def _drain():
            if not pending:
                return
            da = _require_dask()
            flat = [d for pair in pending for d in pair]
            done = da.compute(*flat)
            for i in range(0, len(done), 2):
                buf_X.append(np.asarray(done[i]))
                buf_y.append(np.asarray(done[i + 1]))
                chunk_sizes.append(np.asarray(done[i]).shape[0])
            pending.clear()

        def _flush(buf_X, buf_y, chunk_sizes):
            _require_torch()
            X_buf = np.concatenate(buf_X, axis=0)
            y_buf = np.concatenate(buf_y, axis=0)
            perm = rng.permutation(len(X_buf))
            X_buf, y_buf = X_buf[perm], y_buf[perm]
            start = 0
            for size in chunk_sizes:
                yield (
                    torch.from_numpy(X_buf[start:start + size]).to(torch.float32),
                    torch.from_numpy(y_buf[start:start + size]).to(torch.float32)
                )
                start += size

        for X_chunk, y_chunk in chunk_iter:
            pending.append((X_chunk, y_chunk))

            if len(pending) == window_size:
                _drain()
                yield from _flush(buf_X, buf_y, chunk_sizes)
                buf_X, buf_y, chunk_sizes = [], [], []

        _drain()
        if buf_X:
            yield from _flush(buf_X, buf_y, chunk_sizes)

class SparseBatchDataset(Dataset):
    def __init__(self, X_csr, y_cat):
        self.X_csr = X_csr.tocsr() if not isinstance(X_csr, sparse.csr_matrix) else X_csr
        self.n = self.X_csr.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return int(idx)


def _sparse_collate(batch_idx, X_csr, y_cat, row_scale=None):
    _require_torch()
    idx = np.asarray(batch_idx, dtype=np.int64)
    xb = np.asarray(X_csr[idx].toarray(), dtype=np.float64)
    if row_scale is not None:
        xb /= row_scale[idx][:, None]
    xb = torch.from_numpy(xb.astype(np.float32))
    yb = torch.from_numpy(np.asarray(y_cat)[idx]).to(torch.float32)
    return xb, yb

def predict_logits(model, x, batch_size=8192, device=None):
    x = row_robust_scale(x)
    if isinstance(x, sparse.csr_matrix):
        x = sparse.csr_matrix.toarray(x)

    # Serialized so multithreaded callers see identical BLAS/thread-pool
    # state as sequential execution (bit-identical logits).
    with _FORWARD_LOCK:
        if isinstance(model, DenseTorch):
            logits = model.predict_logits(x, batch_size=batch_size, device=device)

        elif isinstance(model, LogisticRegression):
            logits = model.predict_logits(x)

        elif isinstance(model, BoostedTrees):
            logits = model.predict_logits(x)

        elif isinstance(model, DummyClassifier):
            logits = model.predict_logits(x)

        else:
            raise Exception('Unknown classifier type.')

    return logits

def predict(
        model, x, threshold=-1, monte_carlo: int=None,
        mc_dropout_p: float=0.5,
        batch_size: int=8192,
        mc_max_rows: int=65536,
        device=None):
    if isinstance(model, DummyClassifier):
        # Single-child node: the label is forced regardless of features, so
        # skip scaling, densification and feature slicing entirely.
        return model.predict(x)
    x = row_robust_scale(x)
    if isinstance(x, sparse.csr_matrix):
        x = sparse.csr_matrix.toarray(x)

    if monte_carlo is not None:
        # Model calls are serialized (see _FORWARD_LOCK) so parallel
        # callers observe identical numerics to sequential execution.
        with _FORWARD_LOCK:
            if isinstance(model, DenseTorch):
                all_logits = model.predict_logits_mc(
                    x, monte_carlo,
                    dropout_p=mc_dropout_p,
                    batch_size=batch_size,
                    mc_max_rows=mc_max_rows,
                    device=device)

            elif isinstance(model, (LogisticRegression, BoostedTrees)):
                # sklearn/catboost stay on CPU: convert once, then mask the single
                # resident tensor per iteration instead of re-copying every pass.
                # Masking uses torch; deterministic (non-MC) sklearn inference
                # stays torch-free.
                _require_torch()
                dev = resolve_device(device)
                x_arr = np.asarray(x, dtype=np.float32)
                x_t = torch.from_numpy(x_arr).to(dev)
                all_logits = []
                with torch.no_grad():
                    for _ in range(monte_carlo):
                        x_masked = F.dropout(x_t, p=mc_dropout_p, training=True).to('cpu').numpy()
                        all_logits.append(model.predict_logits(x_masked))

                all_logits = np.array(all_logits)

            elif isinstance(model, DummyClassifier):
                return model.predict(x)

            else:
                raise Exception('Unknown classifier type')

        logits = np.mean(all_logits, axis=0)

    else:
        with _FORWARD_LOCK:
            if isinstance(model, DenseTorch):
                logits = model.predict_logits(x, batch_size=batch_size, device=device)

            elif isinstance(model, LogisticRegression):
                logits = model.predict_logits(x)

            elif isinstance(model, BoostedTrees):
                logits = model.predict_logits(x)

            elif isinstance(model, DummyClassifier):
                return model.predict(x)

            else:
                raise Exception('Unknown classifier type')
        
    max_activation = np.max(logits, axis=1)
    pred = np.argmax(logits, axis=1).astype(int)
    pred = np.array([model.labels_dec[p] for p in pred])
    pred[max_activation <= threshold] = ''

    if monte_carlo is not None:
        return pred, all_logits

    return pred

def samples_per_class(y):
    _require_torch()
    spc = list(torch.zeros(y.shape[1]))
    classes_counted = np.unique(np.argmax(y, axis=1), return_counts=True)
    for c, samples in zip(classes_counted[0], classes_counted[1]):
        spc[c] = samples

    return spc

def set_threads(num_threads, parallelize, num_workers=None):
    _require_torch()
    budget = train_resource_budget()
    env_workers = os.environ.get('COMPOCYTE_NUM_WORKERS')
    if env_workers is not None:
        num_workers = max(0, int(env_workers))
    elif num_workers is not None:
        num_workers = max(0, int(num_workers))
    elif parallelize:
        # When training nodes in parallel (CPU pool), each child process owns
        # its DataLoader; using workers inside each child adds overhead.
        num_workers = 0
    else:
        cpu = os.cpu_count() or 4
        num_workers = min(4, max(1, cpu // 8))
    try:
        want_threads = int(num_threads)
    except Exception:
        want_threads = 1
    num_threads = max(1, min(want_threads, budget['max_threads']) - num_workers)

    logger.info(f'num_workers set to {num_workers}')
    torch.set_num_threads(num_threads)
    logger.info(f'num_threads set to {torch.get_num_threads()}')
    return num_workers

def dataloaders_from_dask(x, y, batch_size, num_workers, seed=12345, pin_memory=False):
    _require_torch()
    da = _require_dask()
    total_samples = x.shape[0]

    indices = np.arange(total_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    indices_train = np.sort(indices[:int(np.floor(total_samples * .8))])
    indices_val = np.sort(indices[int(np.floor(total_samples * .8)):])
    # Keep sparse input sparse: dask slices row chunks without densifying and
    # map_blocks(_to_dense) below materializes one batch-size chunk at a time.
    # Densifying here would allocate the full node matrix up front.
    x_d = da.from_array(
        x if sparse.issparse(x) else np.asarray(x),
        chunks=(max(1, batch_size), x.shape[1]))
    y_d = da.from_array(np.asarray(y), chunks=(max(1, batch_size), y.shape[1]))
    x_train, y_train = x_d[indices_train], y_d[indices_train]
    x_val, y_val = x_d[indices_val], y_d[indices_val]

    cfg = _resolve_loader_config(total_samples, x.shape[1], batch_size, num_workers)
    batch_size = min(cfg['batch_size'], x_train.shape[0])
    x_train = x_train.rechunk((batch_size, x_train.shape[1]))
    x_train = x_train.map_blocks(
        _to_dense,
        dtype=np.float32)
    y_train = y_train.rechunk((batch_size, y_train.shape[1]))
    train_dataset = DaskBatchDataset(x_train, y_train)
    num_batches = len(train_dataset.X_chunks)
    train_dataloader = DataLoader(
        train_dataset, batch_size=None, num_workers=cfg['num_workers'],
        pin_memory=bool(pin_memory) and torch.cuda.is_available(),
        **({'prefetch_factor': cfg['prefetch_factor']} if cfg['num_workers'] > 0 else {}))

    batch_size = min(batch_size, x_val.shape[0])
    x_val = x_val.rechunk((batch_size, x_val.shape[1]))
    x_val = x_val.map_blocks(
        _to_dense,
        dtype=np.float32)
    y_val = y_val.rechunk((batch_size, y_val.shape[1]))
    val_dataset = DaskBatchDataset(x_val, y_val)
    num_batches_val = len(val_dataset.X_chunks)
    val_dataloader = DataLoader(
        val_dataset, batch_size=None, num_workers=cfg['num_workers'],
        pin_memory=bool(pin_memory) and torch.cuda.is_available(),
        **({'prefetch_factor': cfg['prefetch_factor']} if cfg['num_workers'] > 0 else {}))

    return train_dataloader, val_dataloader, num_batches, num_batches_val

def _seeded_split(dataset, seed):
    _require_torch()
    gen = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [0.8, 0.2], generator=gen)

def _loader_kwargs(workers, prefetch, pin):
    kw = {'num_workers': workers, 'pin_memory': bool(pin) and torch.cuda.is_available()}
    if workers > 0:
        kw['prefetch_factor'] = prefetch
        kw['persistent_workers'] = True
    return kw

def dataloaders_from_dense(x, y, batch_size, num_workers, seed=12345, pin_memory=False,
                           row_scale=None):
    _require_torch()
    n_cells = x.shape[0]
    n_genes = x.shape[1]
    cfg = _resolve_loader_config(n_cells, n_genes, batch_size, num_workers)
    batch_size, workers = cfg['batch_size'], cfg['num_workers']
    sparse_kw = _loader_kwargs(workers, cfg['prefetch_factor'], pin_memory)

    if sparse.issparse(x):
        dataset = SparseBatchDataset(x, y)
        collate = partial(_sparse_collate, X_csr=dataset.X_csr, y_cat=np.asarray(y),
                          row_scale=row_scale)
        train_dataset, val_dataset = _seeded_split(dataset, seed)
        batch_size = min(batch_size, len(train_dataset))
        leaves_remainder = len(train_dataset) % batch_size == 1
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=leaves_remainder,
            collate_fn=collate,
            **sparse_kw)

        batch_size = min(batch_size, len(val_dataset))
        leaves_remainder = len(val_dataset) % batch_size == 1
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=leaves_remainder,
            collate_fn=collate,
            **sparse_kw)

        return train_dataloader, val_dataloader

    x = torch.from_numpy(np.asarray(x)).to(torch.float32)
    y = torch.from_numpy(np.asarray(y)).to(torch.float32)
    dataset = TensorDataset(x, y)
    train_dataset, val_dataset = _seeded_split(dataset, seed)
    batch_size = min(batch_size, len(train_dataset))
    leaves_remainder = len(train_dataset) % batch_size == 1
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=leaves_remainder,
        **sparse_kw)

    batch_size = min(batch_size, len(val_dataset))
    leaves_remainder = len(val_dataset) % batch_size == 1
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=leaves_remainder,
        **sparse_kw)

    return train_dataloader, val_dataloader

def fit_torch(
        model: DenseTorch,
        x: np.array, y: np.array,
        epochs: int=40, batch_size: int=64,
        starting_lr: float=0.01, max_lr: float=0.1, momentum: float=0.5,
        parallelize: bool=True, num_threads: int=1, num_workers: int=None,
        beta: float=0.8, gamma: float=2.0, class_balance: bool=True, max_cells: int=1_000_000,
        device=None, seed: int=12345, resource_stats: dict=None,
        row_scale: np.ndarray=None):

    _require_torch()
    num_workers = set_threads(num_threads, parallelize, num_workers)
    if seed is not None:
        torch.manual_seed(int(seed))
    dev = resolve_device(device or getattr(model, 'device', None))
    if hasattr(model, 'to_device'):
        model.to_device(dev)
    else:
        model.to(dev)
    y = to_categorical(y, num_classes=len(model.labels_enc.keys()))    
    total_samples = x.shape[0]

    # GPU optimization: for small node matrices move the whole tensor to the
    # device once, eliminating per-batch host->device copies. For larger
    # matrices fall back to host-resident data with fork-safe CPU workers that
    # prefetch/pin batches. CPU callers stay sparse to keep memory low.
    _GPU_DENSE_LIMIT_BYTES = 1024 * 1024 ** 2
    tensors_on_device = False
    pin = False
    if str(dev.type) == 'cuda':
        dense_bytes = x.shape[0] * x.shape[1] * 4
        y_t = torch.from_numpy(y).to(dev, non_blocking=True)
        if dense_bytes <= _GPU_DENSE_LIMIT_BYTES:
            x_dense = np.asarray(_to_dense(x), dtype=np.float64)
            if row_scale is not None:
                x_dense = x_dense / row_scale[:, None]
            x_t = torch.from_numpy(x_dense.astype(np.float32)).to(dev, non_blocking=True)
            tensors_on_device = True
            dataset = TensorDataset(x_t, y_t)
            train_dataset, val_dataset = _seeded_split(dataset, seed)
            # No workers/CUDA memory sharing, no host pin overhead.
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                drop_last=(len(train_dataset) % batch_size == 1))
            batch_size = min(batch_size, len(val_dataset))
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                drop_last=(len(val_dataset) % batch_size == 1))
            num_batches = len(train_dataloader)
            num_batches_val = len(val_dataloader)
            if resource_stats is not None:
                resource_stats['tensors_on_device'] = True
                resource_stats['dense_bytes'] = dense_bytes
        else:
            pin = (os.environ.get('COMPOCYTE_PIN_MEMORY', '1') == '1')

    if not tensors_on_device:
        pin = (str(dev.type) == 'cuda'
               and os.environ.get('COMPOCYTE_PIN_MEMORY', '1') == '1')
        # Sparse matrices are kept in-memory and served by SparseBatchDataset,
        # which only densifies one batch at a time. Dask is only useful for
        # genuine out-of-core dense arrays that exceed max_cells.
        if total_samples > max_cells and not sparse.issparse(x):
            train_dataloader, val_dataloader, num_batches, num_batches_val = dataloaders_from_dask(
                x, y, batch_size, num_workers, seed=seed, pin_memory=pin)

        else:
            train_dataloader, val_dataloader = dataloaders_from_dense(
                x, y, batch_size, num_workers, seed=seed, pin_memory=pin,
                row_scale=row_scale)
            num_batches = len(train_dataloader)
            num_batches_val = len(val_dataloader)

    if resource_stats is not None:
        cfg = _resolve_loader_config(total_samples, x.shape[1], batch_size,
                                     0 if tensors_on_device else num_workers)
        resource_stats.update(cfg)
        resource_stats['dataloader_path'] = 'device' if tensors_on_device else (
            'dask' if total_samples > max_cells else 'dense')
        resource_stats['n_samples'] = int(total_samples)
        resource_stats['max_cells'] = int(max_cells)
        if row_scale is not None:
            resource_stats['row_scale_applied'] = True
        resource_stats['device'] = str(dev)
        resource_stats['pin_memory'] = bool(pin)
        resource_stats['num_workers_requested'] = 0 if tensors_on_device else num_workers
        resource_stats['torch_threads'] = torch.get_num_threads()
        resource_stats['tensors_on_device'] = tensors_on_device
        resource_stats['rss_before_bytes'] = current_rss_bytes()
        resource_stats['rss_peak_before_bytes'] = peak_rss_bytes()
        if str(dev.type) == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    t_fit0 = time.perf_counter()
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=starting_lr, 
        momentum=momentum
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        div_factor=10,
        epochs=epochs,
        steps_per_epoch=num_batches
    )
    loss_function = _require_balanced_loss()(
        loss_type="focal_loss",
        samples_per_class=samples_per_class(y),
        beta=beta, # class-balanced loss beta
        fl_gamma=gamma, # focal loss gamma
        class_balanced=class_balance,
        safe=True
    )
    best_state = None
    best_val = np.inf
    learning_curve = pd.DataFrame(columns=['loss', 'val_loss', 'lr'])
    for epoch in range(epochs):
        if hasattr(train_dataloader.dataset, 'set_epoch'):
            train_dataloader.dataset.set_epoch(epoch)
            
        model.train()
        cumulative_loss = 0
        for xb, yb in train_dataloader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            logits = model(xb)
            logits = torch.clamp(logits, 0, 1)
            loss = loss_function(logits, torch.argmax(yb, dim=-1).to(torch.int64))
            loss.backward()

            cumulative_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        cumulative_loss = cumulative_loss / num_batches
        model.eval()
        running_vloss = 0.0    
        if hasattr(val_dataloader.dataset, 'set_epoch'):
            val_dataloader.dataset.set_epoch(epoch)

        for xb, yb in val_dataloader:                
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            logits = model(xb)
            logits = torch.clamp(logits, 0, 1)
            val_loss = loss_function(logits, torch.argmax(yb, dim=-1).to(torch.int64)).item()
            running_vloss += val_loss

        val_loss = running_vloss / num_batches_val
        learning_curve.loc[epoch, ['loss', 'val_loss', 'lr']] = cumulative_loss, val_loss, scheduler.get_last_lr()
        # Keep only the best snapshot: v1 retained all epochs (O(epochs * params) RAM).
        # Strict '<' reproduces np.argmin, which selects the first best epoch.
        if best_state is None or val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.is_fitted = True
    if resource_stats is not None:
        resource_stats['fit_s'] = time.perf_counter() - t_fit0
        resource_stats['rss_after_bytes'] = current_rss_bytes()
        resource_stats['rss_delta_bytes'] = max(
            0, resource_stats['rss_after_bytes']
            - resource_stats.get('rss_before_bytes', 0))
        resource_stats['n_batches'] = int(num_batches)
        if str(dev.type) == 'cuda' and torch.cuda.is_available():
            resource_stats['gpu_peak_bytes'] = int(torch.cuda.max_memory_allocated())

    return learning_curve

def fit_logreg(model: LogisticRegression, x, y, **fit_kwargs):
    fit = model.model.fit(
        x, y,
    )
    model.is_fitted = True
    #model.labels_enc = {label: i for i, label in enumerate(model.model.classes_)}
    #model.labels_dec = {model.labels_enc[label]: label for label in model.labels_enc.keys()}

    return fit

def fit_trees(model: BoostedTrees, x, y, **fit_kwargs):
    x, x_val, y, y_val = train_test_split(x, y, train_size=0.75, random_state=42)
    if not np.all(np.isin(np.unique(y_val), np.unique(y))):
        # if the validation set contains labels not in the training set, remove them
        x_val = x_val[np.isin(y_val, np.unique(y))]
        y_val = np.array([label for label in y_val if label in np.unique(y)])

    fit = model.model.fit(
        x, y,
        eval_set=[(x_val, y_val)],
        #**fit_kwargs
    )
    model.is_fitted = True

    return fit

def fit(
        model: Union[DenseTorch, LogisticRegression, DummyClassifier],
        x: np.array, y: np.array,
        standardize_idx: list=None,
        **fit_kwargs):
    """Args:
        model (Union[DenseTorch, LogisticRegression, DummyClassifier]): Model to be fitted.
        x (np.array): Input data.
        y (np.array): Target data in the shape of a 1-dimensional array of label strings.

    Returns:
        _type_: _description_
    """
    if isinstance(model, DummyClassifier):
        # Single-child node: nothing to learn from features.
        model.is_fitted = True
        return model.fit(x, y)

    # Standardize batches separately if list of idxs per dataset is provided
    if standardize_idx is not None:
        for idx in standardize_idx:
            scaled = row_robust_scale(x[idx])
            x[idx] = sparse.csr_matrix(scaled) if sparse.issparse(x) else scaled
    elif isinstance(model, DenseTorch) and sparse.issparse(x):
        # Keep the training matrix sparse so no full dense copy is materialized.
        # For large-node training, defer the scale application into the batch
        # collate to avoid a full scaled sparse copy per worker process.
        row_scale = row_robust_scale_sparse_to_scale(x)
        fit_kwargs['row_scale'] = row_scale
    else:
        x = row_robust_scale(x)


    if not isinstance(model, DenseTorch):
        x = _to_dense(x)

    y = np.array([model.labels_enc[label] for label in y])
    if isinstance(model, DenseTorch):
        return fit_torch(model, x, y, **fit_kwargs)

    elif isinstance(model, LogisticRegression):
        return fit_logreg(model, x, y, **fit_kwargs)

    elif isinstance(model, BoostedTrees):
        return fit_trees(model, x, y, **fit_kwargs)

    elif isinstance(model, DummyClassifier):
        model.is_fitted = True
        return model.fit(x, y)