from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.pretrained import til_pretrained, pbmc_pretrained


def test_til_pretrained():
    hc = til_pretrained()
    assert isinstance(hc, HierarchicalClassifier)
    assert hasattr(hc, 'graph')

def test_pbmc_pretrained():
    hc = pbmc_pretrained()
    assert isinstance(hc, HierarchicalClassifier)
    assert hasattr(hc, 'graph')