import celltypist
import numpy as np
import pandas as pd
from scanpy import AnnData
from scipy.sparse.csr import csr_matrix

class CellTypistWrapper():
    """Add explanation.
    """

    def __init__(self, **kwargs):
        """Add explanation.
        """

        pass

    def train(self, x, y, **kwargs):
        x = x.copy()
        self.model = celltypist.train(X=x, labels=y)

    def validate(self, x, y, **kwargs):
        return (0, 0)

    def predict(self, x):
        x = x.copy()
        pred = celltypist.annotate(x, model=self.model, majority_voting=True)
        pred_adata = pred.to_adata()

        return np.array(pred_adata.obs['majority_voting'])