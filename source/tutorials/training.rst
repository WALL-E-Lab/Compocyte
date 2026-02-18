Training custom classifiers
===========================

This tutorial outlines a minimal training loop for local classifiers used in a
hierarchical setup.

1. Build or load an `AnnData` with appropriate `obs` columns describing labels.
2. Use `HierarchicalClassifier.run_feature_selection(node, ...)` to select features.
3. Create a local classifier with `create_local_classifier(node, classifier_type=...)`.
4. Train with `train_single_node(node, ...)`.

Example:

.. code-block:: python

   clf.run_feature_selection('Root')
   clf.create_local_classifier('Root', classifier_type='DenseTorch')
   clf.train_single_node('Root', epochs=10, batch_size=32)

For hyperparameter tuning see `Compocyte.core.tuner`.
