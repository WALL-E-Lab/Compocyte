.. Compocyte documentation master file

=====================================
Compocyte Documentation
=====================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   readme

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/quick_start
   tutorials/training_classifiers
   tutorials/hierarchical_inference

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Resources

   troubleshooting
   resources/citation
   GitHub Repository <https://github.com/WALL-E-Lab/Compocyte>

Welcome to Compocyte
====================

Compocyte is a composite classifier for modular hierarchical cell type annotation of single cell data.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Getting Started
      :link: readme
      :link-type: doc

      Learn how to install and use Compocyte with our quickstart guide.

   .. grid-item-card:: Tutorials
      :link: tutorials/quick_start
      :link-type: doc

      Walk through practical examples and learn best practices.

   .. grid-item-card:: API Reference
      :link: api/modules
      :link-type: doc

      Complete documentation of all Compocyte modules and functions.

   .. grid-item-card:: Troubleshooting
      :link: troubleshooting
      :link-type: doc

      Solutions to common issues and frequently asked questions.

   .. grid-item-card:: GitHub
      :link: https://github.com/WALL-E-Lab/Compocyte

      View source code, report issues, and contribute.

Key Features
============

- **Hierarchical Classification**: Build modular classifiers following a local classifier per parent node (LCPN) approach
- **Flexible Models**: Use PyTorch, scikit-learn, or CatBoost as your local classifiers
- **Interactive Annotation**: Easily modify individual classifiers for human-in-the-loop improvements
- **Multi-modal Support**: Works with single cell RNA-seq and other modalities compatible with AnnData and scanpy

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
