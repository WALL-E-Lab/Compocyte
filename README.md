# Compocyte

<strong>Compocyte</strong> is a composite classifier for modular hierarchical cell type annotation of single cell data. Using Compocyte you can build different hierarchical classifier architectures following a local classifier per parent node approach. Local classifiers are built around pytorch, sklearn or CatBoost. Local classifiers can be individually modified to account for alterations in classification taxonomies or selectively improve specific annotations in human-in-the-loop approaches. While compocyte has been primarily developed for single cell RNA sequencing data it can also be used with other single cell data compatible with the AnnData and scanpy packages.

<br>

	**Please cite the Compocyte preprint: https://doi.org/10.64898/2026.05.30.728980**

## Installation

For most users, we suggest making use of our prepared Docker image. This comes with all required dependencies for standard use cases.

### Docker

1. Pull the image.
```bash
docker pull chbeltz/compocyte:latest
```
2. Start an interactive container.
```bash
docker run -it --rm chbeltz/compocyte:latest bash
```

If you want to work with your own data, mount a local directory into the container:
```bash
docker run -it --rm \
  -v /path/to/your/data:/data \
  chbeltz/compocyte:latest bash
```
Your files will then be accessible inside the container at /data.

### Manual installation

Alternatively, you can install Python 3.14 using micromamba or another environment manager, and then install Compocyte and its dependencies from conda-forge/PyPI.

```bash
micromamba create -n compocyte_python314 python=3.14
micromamba activate compocyte_python314
micromamba install gcc gxx graphviz pygraphviz
pip install Compocyte
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

## Getting started

For a quick dive into using Compocyte to label single-cell data, refer to our [Getting started](https://compocyte.readthedocs.io/en/latest/tutorials/getting_started.html) on readthedocs. There, you can also find tutorials to help you make full use of Compocyte's features.

## Citation

When using our pretrained classification models, please cite the Zenodo publications above.

When using Compocyte, please cite our publication (DOI will be provided shortly).
