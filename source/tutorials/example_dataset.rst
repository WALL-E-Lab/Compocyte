Example dataset and end-to-end example
=====================================

This page demonstrates creating a tiny synthetic dataset and running the main
inference flow. Use this as a minimal reproducible example for testing.

.. code-block:: python

   import numpy as np
   import scanpy as sc
   from anndata import AnnData
   from Compocyte.core.hierarchical_classifier import HierarchicalClassifier

   # Create tiny synthetic data (10 cells, 5 genes)
   X = np.random.poisson(1.0, size=(10, 5)).astype(float)
   obs = {"cell_type": ["A"]*5 + ["B"]*5}
   var = {"gene_ids": [f"g{i}" for i in range(5)]}
   adata = AnnData(X=X)
   adata.obs["cell_type"] = obs["cell_type"]

   # Initialize classifier (no models present; this is only to exercise API)
   hc = HierarchicalClassifier(save_path="./tmp_model")
   hc.load_adata(adata)

   # This example won't train models — it shows usage of API methods
   try:
       hc.predict_all_child_nodes('Root')
   except Exception as e:
       print('Expected error (no models):', e)

Notes
-----

- For realistic analyses use real single-cell `h5ad` files and train or load
  pretrained models before calling `predict_all_child_nodes()`.
