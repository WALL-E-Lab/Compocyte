# Compocyte 🐙 🎯

<img src="https://github.com/WALL-E-Lab/Compocyte/blob/main/Compocyte.png" alt="Logo" width="250" align="left" hspace="20"/>

<strong>Compocyte</strong>
is a composite classifier for modular hierarchical cell type annotation of single cell data. Using Compocyte you can build different hierarchical classifier architectures (local classifier per parent node, local classifer per node and local classifier per level) using all relevant models from pytorch, TensorFlow and keras. Local classifiers can be individually modified to account for alterations in classification taxonomies or selectively improve specific annotations in human-in-the-loop approaches. While compocyte has been primarily developed for single cell RNA sequencing data it can also be used with other single cell data compatible with the AnnData and scanpy packages.<br>

<br clear="all" />

## Getting started

### Installation

#### Install dependencies

All dependencies are included in the compocyte Docker image uploaded to Docker Hub. Alternatively, you can install Python 3.14 and then install Compocyte from source and its dependencies.

```
micromamba create -n compocyte_python314 python=3.14
micromamba activate compocyte_python314
micromamba install catboost
pip install "git+https://github.com/WALL-E-Lab/Compocyte.git"
```

### Inference 

Hierarchically classify cells using a pretrained classifier

```
import Compocyte
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.pretrained import til_pretrained, pbmc_pretrained

hc = til_pretrained()
adata = Compocyte.data.test_data()
hc.load_adata(adata)
hc.predict_all_child_nodes('blood')
print(hc.adata.obs)
```

## Pretrained model files

Pretrained Compocyte models are available on Zenodo.
- [PBMC classifier 1.0](https://zenodo.org/records/19708295)
- [TIL classifier 1.0](https://zenodo.org/records/19707910)

They can also be loaded from within Compocyte the following way:
```
import Compocyte
pbmc_hc = Compocyte.pretrained.pbmc_pretrained()
til_hc = Compocyte.pretrained.til_pretrained()
```

## Full tutorials

For step-by-step guides and explanations to using Compocyte please refer to one of our tutorials.

- [Applying a pre-trained classifier]()
- [Training a custom classifier](https://colab.research.google.com/drive/10Jht-iKHjGbwOnoMqeiIHrqPA37kO745?usp=sharing)
- [Expanding an existing classifier]()
- [Importing an existing subset classifier]()

## Citation

Results will soon be published.
