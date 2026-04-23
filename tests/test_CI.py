import Compocyte
import anndata
from Compocyte.core.tools import dict_depth

def test_CI():
    assert True

def test_sample_data():
    adata = Compocyte.data.sample_data()
    assert isinstance(adata, anndata.AnnData)
    assert len(adata.obs_names) == 861
    assert len(adata.var_names) == 1011

def test_sample_hierarchy():
    hierarchy = Compocyte.data.sample_hierarchy()
    assert isinstance(hierarchy, dict)
    assert dict_depth(hierarchy) == 3
    assert 'T_cell' in hierarchy