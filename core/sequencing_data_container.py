from .tools import is_counts

class SequencingDataContainer():
	"""Add explanation
	"""

	def __init__(self, adata, batch_key='batch'):
		self.adata = adata
        self.batch_key = batch_key
		self.ensure_not_view()
		self.check_for_counts()
		self.ensure_batch_assignment()

	def ensure_not_view(self):
		"""Ensure that the AnnData object saved within is not a view.
		"""

		if adata.is_view:
            self.adata = adata.copy()

        else:
            self.adata = adata

    def ensure_batch_assignment(self):
    	"""Ensure that the batch_key supplied is actually a key in adata.obs.
    	"""

    	if not self.batch_key in self.adata.obs.columns:
    		raise KeyError('The batch key supplied does not match any column in adata.obs.')

    def check_for_counts(self):
        """Checks adata.X and adata.raw.X for presence of raw count data.
        """

        if is_counts(self.adata.X):
            pass

        else:
            if hasattr(self.adata, 'raw') and self.adata.raw != None and is_counts(self.adata.raw.X):
                pass

            else:
                raise ValueError('No raw counts found in adata.X or adata.raw.X.')