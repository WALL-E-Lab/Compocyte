# Compocyte 🐙 🎯
    
<div align="center">
<img src="https://github.com/WALL-E-Lab/Compocyte/blob/main/Compocyte.png" alt="Logo" width="150" />
</div>
<br>

<strong>Compocyte</strong> is a composite classifier for modular hierarchical cell type annotation of single cell data. Using Compocyte you can build different hierarchical classifier architectures (local classifier per parent node, local classifer per node and local classifier per level) using all relevant models from pytorch, TensorFlow and keras. Local classifiers can be individually modified to account for alterations in classification taxonomies or selectively improve specific annotations in human-in-the-loop approaches. While compocyte has been primarily developed for single cell RNA sequencing data it can also be used with other single cell data compatible with the AnnData and scanpy packages.


<br clear="all" />

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Getting started](#getting-started)
  - [Installation](#installation)
    - [Docker](#docker)
    - [Manual installation](#manual-installation)
  - [Pretrained model files](#pretrained-model-files)
    - [Inference](#inference)
- [Citation](#citation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Getting started

## Installation

For most users, we suggest making use of our prepared Docker image. This comes with all required dependencies for standard use cases.

### Docker

Will be added shortly.

### Manual installation

Alternatively, you can install Python 3.14 using micromamba or another environment manager, and then install Compocyte and its dependencies from source/PyPI.

```
micromamba create -n compocyte_python314 python=3.14
micromamba activate compocyte_python314
micromamba install catboost
pip install "git+https://github.com/WALL-E-Lab/Compocyte.git"
```

## Pretrained model files

Pretrained Compocyte models are available on Zenodo.
- [PBMC classifier 1.0](https://zenodo.org/records/19708295)
- [TIL classifier 1.0](https://zenodo.org/records/19707910)

They can also be loaded from within Compocyte the following way:
```python
import Compocyte
pbmc_hc = Compocyte.pretrained.pbmc_pretrained()
til_hc = Compocyte.pretrained.til_pretrained()
```

### Inference 

You can try out our pretrained models to infer cell type predictions on the included tumor-infiltrating leukocyte test dataset in the following way: 

```python
import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.pretrained import til_pretrained, pbmc_pretrained

hc = til_pretrained()
adata = Compocyte.data.sample_data()
hc.load_adata(adata)

hc.predict_all_child_nodes('blood')
print(hc.adata.obs)
```
Because the prediction process is hierarchical in nature we need to specify the root node for our inference run.  Don't be confused by our choice of root node above. The fact that the TIL hierarchy starts with "blood" will be patched in future version.

Alternatively you can do the same on the sample PBMC dataset included in scanpy.

```python
import scanpy as sc

hc = pbmc_pretrained()
adata = sc.datasets.pbmc3k()
hc.load_adata(adata)
hc.predict_all_child_nodes('Blood')
print(hc.adata.obs)
```

# Citation

When using our pretrained classification models, please cite the Zenodo publications above.

When using Compocyte, please cite our publication (DOI will be provided shortly).
