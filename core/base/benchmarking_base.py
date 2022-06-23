class BenchmarkingBase():
    """Add explanation.
    """

    def get_last_annotation(self, obs_names=None, barcodes=None, pred_enough=False, true_enough=False):
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
                if !pred_enough:
                    obs_df_level = obs_df_level[obs_df_level[true_key].isin([np.nan, '', 'nan']) != True]

                if !true_enough:
                    obs_df_level = obs_df_level[obs_df_level[pred_key].isin([np.nan, '', 'nan']) != True]
                obs_df_level.rename(columns={true_key: 'true_last', pred_key: 'pred_last'}, inplace=True)   
                obs_df_level = obs_df_level.astype(str)
                level_barcodes = [x for x in obs_df_level.index if x in obs_df.index]
                obs_df.loc[level_barcodes, ['true_last', 'pred_last']] = obs_df_level.loc[level_barcodes, ['true_last', 'pred_last']]

        return obs_df

    def amount_early_stopping(self, obs_names=None, value='pct'):
        if obs_names is None:
            obs_names = self.obs_names

        leaf_nodes = self.get_leaf_nodes()
        obs_df = self.get_last_annotation(obs_names, barcodes, pred_enough=True)
        n_total = len(obs_df)
        stopped_early_df = obs_df[obs_df['pred_last'].isin(leaf_nodes) != True]
        n_stopped_early = len(stopped_early_df)
        amount = n_stopped_early / n_total
        if value == 'pct':
            amount = round(amount * 100, 2)

        label_to_degree_early_stopping = {}
        for label in stopped_early_df['pred_last'].unique():
            avg_distance = self.degree_early_stopping(label)
            label_to_degree_early_stopping[label] = avg_distance

        stopped_early_df.obs['degree_early_stopping'] = stopped_early_df.obs['pred_last'].map(label_to_degree_early_stopping)
        degree = np.mean(stopped_early_df.obs['degree_early_stopping'].values)

        return amount, degree

    def degree_early_stopping(self, node):
        child_nodes = [n for n in nx.traversal.bfs_tree(self.graph, node) if n != node]
        leaf_nodes = [x for x in self.graph.nodes() \
                        if self.graph.out_degree(x) == 0 \
                        and self.graph.in_degree(x) == 1]
        child_leaf_nodes = [x for x in leaf_nodes if x in child_nodes]
        distances = []
        for cln in child_leaf_nodes:
            diff_levels = len(nx.shortest_path(self.graph, node, cln)) - 1
            distances.append(diff_levels)

        if len(distances) == 0:
            raise Exception('There should be child leaf nodes if cells are marked \
            as having sotpped early')

        avg_distance = sum(distances) / len(distances)
        return avg_distance

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