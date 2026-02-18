Getting started with Compocyte
===============================

This short tutorial shows how to load an AnnData object and run a pretrained
hierarchical classifier to annotate cells.

1. Install dependencies (see project README for full instructions).
2. Install quick docs dependencies (optional, for running docs examples):

.. code-block:: bash

   pip install scanpy anndata

3. Load your `adata` file with `scanpy`:

   .. code-block:: python

      import scanpy as sc
      adata = sc.read('path/to/adata.h5ad')

3. Load a pretrained Compocyte classifier and attach data:

   .. code-block:: python

      from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
      clf = HierarchicalClassifier('path/to/model/folder')
      clf.load()
      clf.load_adata(adata)

4. Predict on the full hierarchy:

   .. code-block:: python

      clf.predict_all_child_nodes('RootCellType')

5. Save annotated `adata`:

   .. code-block:: python

      clf.adata.write('annotated_adata.h5ad')

Screenshot
----------

If you want to include quick visuals in the docs, the project logo is available
and used in the documentation theme. You can also include result screenshots
here by placing images in `source/_static` and referencing them, e.g.:

.. image:: /_static/Compocyte.png
   :alt: Compocyte logo
   :width: 300px

