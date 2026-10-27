"""Plugin discovery for Compocyte local classifiers.

Classifiers are resolved by name through
``[project.entry-points."compocyte.classifiers"]`` so optional backends
(CatBoost today; XGBoost/JAX tomorrow) never need to be imported until they
are actually instantiated. Built-in classifiers always resolve, even on
minimal installs, because their modules import cleanly without optional
third-party packages (heavy imports live behind per-module guards).

Usage:
    from Compocyte.core.models.registry import get
    cls = get('boosted_trees')   # -> BoostedTrees class (instantiation may
                                 #    still require pip install "Compocyte[boosted]")
"""
import importlib
import importlib.metadata as _metadata
import re
import warnings

ENTRY_POINT_GROUP = "compocyte.classifiers"

# Fallback mapping used when package metadata is unavailable (e.g. running
# from an uninstalled source tree). Values are lazy "module:attr" references
# so importing this module never imports a backend.
_BUILTIN_FALLBACK = {
    "dense_torch": "Compocyte.core.models.dense_torch:DenseTorch",
    "logistic_regression": "Compocyte.core.models.log_reg:LogisticRegression",
    "dummy": "Compocyte.core.models.dummy_classifier:DummyClassifier",
    "boosted_trees": "Compocyte.core.models.trees:BoostedTrees",
}

_EXTRA_HINTS = {
    "dense_torch": 'pip install "Compocyte[torch]"',
    "boosted_trees": 'pip install "Compocyte[boosted]"',
}


def _normalize(name):
    # Accepts historical CamelCase spellings ('DenseTorch'), snake_case,
    # dashes and surrounding whitespace. Consecutive capitals ('DUMMY')
    # are treated as one word.
    text = str(name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return re.sub(r"[\s\-_]+", "_", text).lower()


def _load_reference(ref):
    module_name, _, attr = ref.partition(":")
    if not module_name or not attr:
        raise ValueError(f"Malformed classifier reference {ref!r}; "
                         "expected 'module:ClassName'.")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(
            f"Classifier entry point {ref!r} does not expose {attr!r}.") from e


def available():
    """Return the sorted names of discoverable classifiers."""
    names = set(_BUILTIN_FALLBACK)
    try:
        eps = _metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:
        eps = ()
    for ep in eps:
        names.add(_normalize(ep.name))
    return sorted(names)


def get(name):
    """Resolve a classifier *class* by name without instantiating it.

    Raises:
        KeyError: if no classifier is registered under ``name``.
        ImportError: if resolving it fails (e.g. optional backend missing);
            the message names the pip extra that provides it when known.
    """
    key = _normalize(name)
    try:
        eps = _metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:
        eps = ()
    for ep in eps:
        if _normalize(ep.name) == key:
            try:
                return ep.load()
            except ImportError as e:
                hint = _EXTRA_HINTS.get(key)
                raise ImportError(
                    f"Cannot load classifier {name!r}."
                    + (f" Install it with: {hint}" if hint else "")) from e
    if key in _BUILTIN_FALLBACK:
        try:
            return _load_reference(_BUILTIN_FALLBACK[key])
        except ImportError as e:
            hint = _EXTRA_HINTS.get(key)
            raise ImportError(
                f"Cannot load classifier {name!r}."
                + (f" Install it with: {hint}" if hint else "")) from e
    raise KeyError(
        f"Unknown classifier {name!r}. Available: {', '.join(available())}")
