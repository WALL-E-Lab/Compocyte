from importlib.resources import files
import anndata as ad
import pickle

def sample_data() -> ad.AnnData:
    """
    Small tumor-infiltrating T cell dataset.
    For testing and tutorials.
    861 cells × 1011 highly variable genes.
    From Beltz et al. (2026).
    
    Returns
    -------
    AnnData with raw counts in .X, cell type labels in .obs['labels']
    """
    data_path = files("Compocyte.data.data").joinpath("test_data.h5ad")
    return ad.read_h5ad(data_path)

def sample_hierarchy() -> dict:
    """
    Test hierarchy for the test dataset.
    
    Returns
    -------
    AnnData with raw counts in .X, cell type labels in .obs['labels']
    """
    data_path = files("Compocyte.data.data").joinpath("test_hierarchy.pkl")
    with open(data_path, 'rb') as f:
        hierarchy = pickle.load(f)
        
    return hierarchy