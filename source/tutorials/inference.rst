Inference with Compocyte
=========================

This tutorial describes typical inference steps using a trained hierarchical
classifier.

- Prepare your `AnnData` and ensure required fields are present in `obs` and `var`.
- Load the `HierarchicalClassifier` as shown in the getting started tutorial.
- Use `predict_all_child_nodes(root_node)` or `predict_node(node)` for targeted
  predictions.

Example:

.. code-block:: python

   clf.predict_node('MyParentNode')

Notes
-----

If predictions are probabilistic, consider storing both predicted labels and
probabilities in `adata.obs` for downstream analysis.
