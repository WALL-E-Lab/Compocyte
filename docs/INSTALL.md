# Installing Compocyte

Bare `pip install Compocyte` is deliberately **minimal**: the importable
package plus sklearn-based (`LogisticRegression`/`DummyClassifier`) inference.
Heavier backends are opt-in extras. Pick the smallest profile that covers
your task:

| Task | Install command | What it adds |
|---|---|---|
| Apply bundled sample data / sklearn inference | `pip install Compocyte` | `anndata`, `scanpy`, `scikit-learn`, `pooch` |
| Apply or train `DenseTorch` models (incl. all pretrained classifiers) | `pip install "Compocyte[torch]"` | `torch` |
| Train `BoostedTrees` nodes | `pip install "Compocyte[boosted]"` | `catboost` |
| Class-balanced focal loss in `fit_torch` | `pip install "Compocyte[focal]"` | `balanced-loss` |
| Ontology knowledge base / custom hierarchies | `pip install "Compocyte[trees]"` | `cytopus` |
| Atlas-scale helpers, Leiden clustering, hierarchy plots | `pip install "Compocyte[atlas]"` | `dask`, `leidenalg`, `pygraphviz` (needs system `graphviz-dev`) |
| Colab tutorials with metacells | `pip install "Compocyte[tutorials]"` | `SEACells` |
| Everything (old default; Docker/CI) | `pip install "Compocyte[all]"` | all of the above + `dev` |
| Contributing / running tests | `pip install -e ".[dev,all]"` | `pytest`, `nbmake` |

## Missing-backend errors

Instantiating a classifier whose backend is not installed raises an
`ImportError` naming the exact extra, e.g.:

```
ImportError: BoostedTrees requires the 'boosted' extra: pip install "Compocyte[boosted]"
```

`import Compocyte` itself always succeeds on a minimal install; only touching
a missing backend raises.

## Plugin classifiers

Third-party classifiers can register under the
`compocyte.classifiers` entry-point group and are then usable by name in
`create_local_classifier(..., classifier_type='<name>')`. See
`src/Compocyte/core/models/registry.py`.
