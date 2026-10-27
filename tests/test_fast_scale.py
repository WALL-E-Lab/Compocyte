import numpy as np
from scipy import sparse
from sklearn.preprocessing import robust_scale
from Compocyte.core.models.fit_methods import row_robust_scale
from Compocyte.data import sample_data


def _ref(X):
    out = robust_scale(X, axis=1, with_centering=False, copy=False,
                       unit_variance=True)
    if sparse.issparse(out):
        out = out.toarray()
    return np.asarray(out)


def _mats():
    rng = np.random.default_rng(7)
    base = sample_data()
    X0 = base.X.toarray().astype(np.float32)
    Xb = np.tile(X0, (12, 1))[:10000]
    keep = rng.random(Xb.shape) < 0.5
    yield sparse.csr_matrix(np.where(keep, Xb, 0).astype(np.float32))
    # constant rows (IQR == 0) and an all-zero row
    Xc = Xb.copy()
    Xc[0] = 3.0
    Xc[1] = 0.0
    yield sparse.csr_matrix(Xc)
    # dense float32 input
    yield np.ascontiguousarray(Xb[:2000])
    # dense float64 input
    yield np.ascontiguousarray(Xb[:2000]).astype(np.float64)


def test_row_robust_scale_bit_identical():
    for X in _mats():
        got = row_robust_scale(X)
        assert np.array_equal(got, _ref(X), equal_nan=True)


def test_row_robust_scale_nan_fallback():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(500, 50)).astype(np.float32)
    X[::37, ::7] = np.nan
    got = row_robust_scale(sparse.csr_matrix(X))
    assert np.array_equal(got, _ref(sparse.csr_matrix(X)), equal_nan=True)
