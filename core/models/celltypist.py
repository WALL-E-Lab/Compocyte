import celltypist
import numpy as np
from scanpy import AnnData

class CellTypistWrapper():
    """Add explanation.
    """

    def __init__(self, **kwargs):
        """Add explanation.
        """

        pass

    def train(self, x, y, **kwargs):
        self.model = celltypist.train(X=x, labels=y, genes=np.array(range(x.shape[1])))

    def validate(self, x, y, **kwargs):
        return (0, 0)

    def predict(self, x):
        adata_validate = AnnData(X=x)
        pred = celltypist.annotate(adata_validate, model=self.model, majority_voting=True)
        pred_adata = pred.to_adata()

        return np.array(pred_adata.obs['majority_voting'])