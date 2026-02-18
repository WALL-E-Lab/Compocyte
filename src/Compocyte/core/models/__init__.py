"""Model implementations used by Compocyte.

This package contains model wrappers and implementations (PyTorch, CatBoost,
logistic regression and simple dummy classifier) used as local classifiers
within hierarchical compositions.
"""

from . import dense_torch, log_reg, dummy_classifier