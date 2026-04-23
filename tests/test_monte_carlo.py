import Compocyte
from Compocyte.pretrained._builtin import til_pretrained


def test_monte_carlo():
    hc = til_pretrained()
    adata = Compocyte.data.sample_data()
    hc.load_adata(adata)
    hc.predict_all_child_nodes('blood', monte_carlo=10)
    print(hc.adata.obs)
    assert 'Level_1_pred' in hc.adata.obs    
    assert 'Level_2_pred' in hc.adata.obs
    assert 'monte_carlo_mean' in hc.adata.obs
    assert 'monte_carlo_std' in hc.adata.obs
    assert not hc.adata.obs['monte_carlo_mean'].isna().all()    
    assert not hc.adata.obs['monte_carlo_std'].isna().all()