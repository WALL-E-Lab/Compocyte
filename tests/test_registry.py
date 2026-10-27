import pytest
from Compocyte.core.models import registry
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.log_reg import LogisticRegression


def test_builtin_names_resolve():
    assert registry.get('dummy') is DummyClassifier
    assert registry.get('logistic_regression') is LogisticRegression
    from Compocyte.core.models.dense_torch import DenseTorch
    assert registry.get('dense_torch') is DenseTorch
    # normalization: CamelCase history, dashes, case, whitespace
    assert registry.get(' Logistic-Regression ') is LogisticRegression
    from Compocyte.core.models.trees import BoostedTrees
    assert registry.get('BoostedTrees') is BoostedTrees


def test_builtin_names_listed():
    names = registry.available()
    for expected in ('dummy', 'logistic_regression', 'dense_torch',
                     'boosted_trees'):
        assert expected in names


def test_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get('no_such_classifier_xyz')


def test_boosted_trees_resolves_to_class():
    # Importable without catboost installed; only instantiation needs it.
    from Compocyte.core.models.trees import BoostedTrees
    assert registry.get('boosted_trees') is BoostedTrees


def test_missing_backend_hint(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'catboost', None)
    from Compocyte.core.models.trees import BoostedTrees
    with pytest.raises(ImportError, match=r"Compocyte\[boosted\]"):
        BoostedTrees(labels=['a', 'b'])
