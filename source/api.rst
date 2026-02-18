API
===

This section contains the auto-generated API reference for the `Compocyte` package.

Core package
------------

.. autosummary::
   :toctree: api
   :template: module.rst

   Compocyte.core
   Compocyte.core.hierarchical_classifier
   Compocyte.core.tools
   Compocyte.core.tuner

Models
------

.. autosummary::
   :toctree: api
   :template: module.rst

   Compocyte.core.models
   Compocyte.core.models.dense_torch
   Compocyte.core.models.log_reg
   Compocyte.core.models.fit_methods
   Compocyte.core.models.dummy_classifier
   Compocyte.core.models.trees

Base classes
------------

.. autosummary::
   :toctree: api
   :template: module.rst

   Compocyte.core.base
   Compocyte.core.base.data_base
   Compocyte.core.base.export_import_base
   Compocyte.core.base.hierarchy_base

If you need a flat list of all modules, see the `auto-generated files <api/index.html>`_.

Quick API index
---------------

Below is a short inline index with one-line summaries for the most important
modules. Click a module name to open its full API page.

* :doc:`api/Compocyte.core` — Core package utilities and primary entry points.
* :doc:`api/Compocyte.core.hierarchical_classifier` — HierarchicalClassifier: orchestrates feature selection, training and hierarchical prediction.
* :doc:`api/Compocyte.core.tools` — Utility functions and small helpers used across the codebase.
* :doc:`api/Compocyte.core.tuner` — Hyperparameter tuning helpers and experiment storage utilities.

Models

* :doc:`api/Compocyte.core.models` — Collection of model wrappers used by the framework.
* :doc:`api/Compocyte.core.models.dense_torch` — PyTorch feed-forward model wrapper and helpers.
* :doc:`api/Compocyte.core.models.log_reg` — Scikit-learn based logistic regression wrapper.
* :doc:`api/Compocyte.core.models.fit_methods` — Shared training/prediction utilities across model wrappers.
* :doc:`api/Compocyte.core.models.trees` — CatBoost wrapped boosted trees implementation.

Base classes

* :doc:`api/Compocyte.core.base` — Base classes used by higher-level modules.
* :doc:`api/Compocyte.core.base.data_base` — Data handling and AnnData helpers.
* :doc:`api/Compocyte.core.base.export_import_base` — Save/load helpers for classifier state and models.
* :doc:`api/Compocyte.core.base.hierarchy_base` — Hierarchy graph management utilities.
