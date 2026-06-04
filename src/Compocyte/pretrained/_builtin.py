import os
import pooch
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier

def til_pretrained() -> HierarchicalClassifier:
    """
    Pretrained TIL classifier from Beltz et al 2026.
    
    Returns:
        :class:`~compocyte.core.hierarchical_classifier.HierarchicalClassifier` with pretrained parameters.

    Example:
        >>> import Compocyte
        >>> from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
        >>> from Compocyte.pretrained import til_pretrained, pbmc_pretrained
        >>> hc = til_pretrained()
        >>> adata = Compocyte.data.sample_data()
        >>> hc.load_adata(adata)
        >>> hc.predict_all_child_nodes('blood')
        >>> print(hc.adata.obs)
    """

    data_path = pooch.retrieve(
        url="https://zenodo.org/records/19707910/files/Compocyte_default_TIL_1.0.tar.gz?download=1",
        known_hash="md5:d88c02b15226742b2a58d9fb34f4c348",
        processor=pooch.Untar(),
    )
    data_path = os.path.dirname(data_path[0])
    hc = HierarchicalClassifier(data_path)
    hc.load()
        
    return hc

def pbmc_pretrained() -> HierarchicalClassifier:
    """
    Pretrained PBMC classifier from Beltz et al 2026.

    Returns:
        :class:`~compocyte.core.hierarchical_classifier.HierarchicalClassifier` with pretrained parameters.

    Example:
        >>> import Compocyte
        >>> import scanpy as sc
        >>> from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
        >>> from Compocyte.pretrained import til_pretrained, pbmc_pretrained
        >>> hc = pbmc_pretrained()
        >>> adata = sc.datasets.pbmc3k()
        >>> hc.load_adata(adata)
        >>> hc.predict_all_child_nodes('Blood')
        >>> print(hc.adata.obs)
    """

    data_path = pooch.retrieve(
        url="https://zenodo.org/records/19708295/files/Compocyte_default_PBMC_1.0.tar.gz?download=1",
        known_hash="md5:c20e549ab213a6bfb38d61d6dc1da7a0",
        processor=pooch.Untar(),
    )
    data_path = os.path.dirname(data_path[0])
    hc = HierarchicalClassifier(data_path)
    hc.load()

    return hc