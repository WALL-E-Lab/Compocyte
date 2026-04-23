import Compocyte
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

def test_til_pretrained_predict():
    hc = til_pretrained()
    adata = Compocyte.data.sample_data()
    hc.load_adata(adata)
    hc.predict_all_child_nodes('blood')
    print(hc.adata.obs)
    assert 'Level_1_pred' in hc.adata.obs    
    assert 'Level_2_pred' in hc.adata.obs

def test_pbmc_pretrained_predict():
    hc = pbmc_pretrained()
    adata = Compocyte.data.sample_data()
    hc.load_adata(adata)
    hc.predict_all_child_nodes('Blood')
    print(hc.adata.obs)
    assert 'Level_1_pred' in hc.adata.obs    
    assert 'Level_2_pred' in hc.adata.obs