.. Compocyte documentation master file

=====================================
Compocyte
=====================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/getting_started
   tutorials/training_PBMC_classifier
   tutorials/modifying_classifiers

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Resources

   resources/troubleshooting
   resources/citation
   GitHub Repository <https://github.com/WALL-E-Lab/Compocyte>

Introduction
==================
Compocyte is a composite classifier for modular hierarchical cell type annotation of single cell data. Using Compocyte you can build different hierarchical classifier architectures following a local classifier per parent node approach. Local classifiers are built around pytorch, sklearn or CatBoost. Local classifiers can be individually modified to account for alterations in classification taxonomies or selectively improve specific annotations in human-in-the-loop approaches. While compocyte has been primarily developed for single cell RNA sequencing data it can also be used with other single cell data compatible with the AnnData and scanpy packages.

Quick links
==================

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: tutorials/getting_started
      :link-type: doc

      Walk through practical examples and learn best practices.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Walk through practical examples and learn best practices.

   .. grid-item-card:: API Reference
      :link: api/modules
      :link-type: doc

      Complete documentation of all Compocyte modules and functions.

   .. grid-item-card:: Troubleshooting
      :link: resources/troubleshooting
      :link-type: doc

      Solutions to common issues and frequently asked questions.

   .. grid-item-card:: GitHub
      :link: https://github.com/WALL-E-Lab/Compocyte

      View source code, report issues, and contribute.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
