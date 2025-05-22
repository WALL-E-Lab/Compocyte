# Compocyte 🐙 🎯

<img src="https://github.com/WALL-E-Lab/Compocyte/blob/main/Compocyte.png" alt="Logo" width="200" align="left" hspace="20"/>

<strong>Compocyte</strong>
is a composite classifier for modular hierarchical cell type annotation of single cell data. Using Compocyte you can build different hierarchical classifier architectures (local classifier per parent node, local classifer per node and local classifier per level) using all relevant models from pytorch, TensorFlow and keras. Local classifiers can be individually modified to account for alterations in classification taxonomies or selectively improve specific annotations in human-in-the-loop approaches. While compocyte has been primarily developed for single cell RNA sequencing data it can also be used with other single cell data compatible with the AnnData and scanpy packages.<br>

<br clear="all" />

## Getting started

### Installation

#### Install dependencies

Tensorflow is currently not available for python >3.12. Here we use micromamba to create a python 3.12 environment first then install the required dependencies.

```
micromamba create -n compocyte_python312 python=3.12
micromamba activate compocyte_python312
micromamba install catboost
pip install torch
micromamba install keras
micromamba install tensorflow
pip install balanced_loss
```

#### Install Compocyte from source

`pip install "git+https://github.com/WALL-E-Lab/Compocyte.git"`

### Inference 

Hierarchically classify cells using a pretrained classifier

```
#paths to model files and adata
model_folder_path = '~/Compocyte/PBMC_pretrained_1.1'
adata_path = '~/adata.h5ad'
adata_save_path = '~/adata_compocyte.h5ad'

#name of root cell type
root_node = 'Blood'

#load adata
adata = sc.read(adata_path)

#load classifier 
classifier= Compocyte.core.hierarchical_classifier.HierarchicalClassifier(model_folder_path)
classifier.load()
classifier.load_adata(adata)

#inference
classifier.predict_all_child_nodes(root_node)

#save adata with classifier outputs
classifier.adata.write(adata_save_path)
print('saved to',adata_save_path)
```

## Pretrained model files

available soon

## Full tutorials

For step-by-step guides and explanations to using Compocyte please refer to one of our tutorials.

- [Applying a pre-trained classifier]()
- [Training a custom classifier](https://colab.research.google.com/drive/1dVSAYgS4yY5ydIA1d-1rqYuIhVAfQI5-?usp=sharing)
- [Expanding an existing classifier]()
- [Importing an existing subset classifier]()

## Citation

Results will soon be published.
