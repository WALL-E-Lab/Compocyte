Getting Started with Compocyte
==============================

.. note::

   This is a placeholder for the quick start tutorial.

   This section will contain:

   - Installation instructions (Docker and manual)
   - Loading sample data
   - Creating your first hierarchical classifier
   - Making predictions
   - Visualizing results

For now, refer to the :doc:`../readme` for installation instructions.

.. code-block:: python

   import Compocyte
   from Compocyte.pretrained import pbmc_pretrained

   # Load a pretrained model
   hc = pbmc_pretrained()

   # Load example data (will be expanded in full tutorial)
   # adata = sc.datasets.pbmc3k()
   # hc.load_adata(adata)
   # predictions = hc.predict_all_child_nodes('Blood')
