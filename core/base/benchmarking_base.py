class BenchmarkingBase():
	"""Add explanation.
	"""

	def get_last_annotation(self, obs_names=None, barcodes=None):
		if obs_names is None:
	        obs_names = self.obs_names

	    if barcodes is None:
	        barcodes = self.adata.obs_names
	        
	    obs_names_pred = [f'{x}_pred' for x in obs_names]
	    for i, (true_key, pred_key) in enumerate(zip(obs_names, obs_names_pred)):
	        if i == 0:
	            obs_df = self.adata.obs.loc[barcodes, [true_key, pred_key]]
	            obs_df = obs_df[obs_df[true_key].isin([np.nan, '', 'nan']) != True]
	            obs_df = obs_df[obs_df[pred_key].isin([np.nan, '', 'nan']) != True]
	            obs_df.rename(columns={true_key: 'true_last', pred_key: 'pred_last'}, inplace=True)
	            obs_df = obs_df.astype(str)		

	        else:
	            obs_df_level = self.adata.obs.loc[barcodes, [true_key, pred_key]]
	            obs_df_level = obs_df_level[obs_df_level[true_key].isin([np.nan, '', 'nan']) != True]
	            obs_df_level = obs_df_level[obs_df_level[pred_key].isin([np.nan, '', 'nan']) != True]
	            obs_df_level.rename(columns={true_key: 'true_last', pred_key: 'pred_last'}, inplace=True)	
	            obs_df_level = obs_df_level.astype(str)
	            level_barcodes = [x for x in obs_df_level.index if x in obs_df.index]
	            obs_df.loc[level_barcodes, ['true_last', 'pred_last']] = obs_df_level.loc[level_barcodes, ['true_last', 'pred_last']]

        return obs_df

	def amount_early_stopping(self, obs_names=None):
		if obs_names is None:
	        obs_names = self.obs_names

		leaf_nodes = self.get_leaf_nodes()
		obs_df = self.get_last_annotation(obs_names, barcodes)

	def amount_missing_annotations(self):
		pass

	def correct_annotations(self, obs_names=None, barcodes=None, value='pct'):
	    """Use the deepest level for which existing annotations and existing predictions both exist to calculate
	    percentage of correctly predicted labels.
	    """

	    obs_df = self.get_last_annotation(obs_names, barcodes)
	    comparison = obs_df['true_last'].values == obs_df['pred_last'].values
	    n_correct = np.where(comparison == True)[0].shape[0]
	    n_total = comparison.shape[0]
	    accuracy = n_correct / n_total
	    if value == 'pct':
	    	accuracy = round(accuracy * 100, 2)

    	return accuracy