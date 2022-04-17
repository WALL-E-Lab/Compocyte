import celltypist
from scanpy import AnnData

class CellTypistWrapper():
	"""Add explanation.
	"""

	def __init__(self, **kwargs):
		"""Add explanation.
		"""

		pass

	def train(self, x, y, **kwargs):
		self.model = celltypist.train(X=x, labels=y)

	def validate(self, x, y, **kwargs):
		return (0, 0)
		adata_validate = AnnData(X=x)
		adata_validate.obs['true_voting'] = y
		pred = celltypist.annotate(adata_validate, model=self.model, majority_voting=True)
    	pred_adata = pred.to_adata()
    	adata_validate.obs['majority_voting'] = pred_adata.obs['majority_voting']