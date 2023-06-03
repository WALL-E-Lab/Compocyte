import gc
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
from scFlorist.core.tools import is_counts, z_transform_properties
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

class DataBase():
    """Add explanation
    """

    def load_adata(
        self,
        adata,
        batch_key='batch'):

        if type(self.adata) != type(None): # Load new adata for transfer learning/prediction
            if batch_key != self.batch_key:
                raise Exception('Batch key must match previously used batch key.')

            self.adata = self.variable_match_adata(adata)
            self.ensure_not_view()
            self.check_for_counts()
            self.ensure_batch_assignment()

        else:
            self.adata = adata
            self.ensure_not_view()
            self.batch_key = batch_key
            self.check_for_counts()
            self.ensure_batch_assignment()

            if hasattr(self, 'hv_genes') and self.hv_genes > 0:
                sc.pp.highly_variable_genes(
                    self.adata, 
                    inplace=True, 
                    flavor='seurat_v3', 
                    n_top_genes=self.hv_genes)
                self.adata = self.adata[:, self.adata.var['highly_variable']]
                self.ensure_not_view()

    def variable_match_adata(
        self, 
        new_adata):

        new_var_names = [v for v in self.adata.var_names if v not in new_adata.var_names]
        if is_counts(new_adata.X):
            pass

        else:
            if hasattr(new_adata, 'raw') and new_adata.raw is not None and is_counts(new_adata.raw.X):
                new_adata.X = new_adata.raw.X

            elif hasattr(new_adata, 'layers') and 'raw' in new_adata.layers and is_counts(new_adata.layers['raw']):
                new_adata.X = new_adata.layers['raw']
                
        if len(new_var_names) > 0:
            new_values = np.empty(
                (len(new_adata.obs_names), len(new_var_names))
            )
            new_values[:] = 0
            new_X = sparse.csr_matrix(
                sparse.hstack(
                    [new_adata.X, sparse.csr_matrix(new_values)]))
            new_var = pd.DataFrame(index=list(new_adata.var_names) + new_var_names)
            new_adata = sc.AnnData(
                X=new_X, 
                var=new_var, 
                obs=new_adata.obs)[:, self.adata.var_names]

        return new_adata

    def add_variables(
        self,
        new_variables):

        new_values = np.empty((
            len(self.adata.obs_names),
            len(new_variables)))
        new_values[:] = 0
        new_X = sparse.csr_matrix(
            sparse.hstack(
                [self.adata.X, sparse.csr_matrix(new_values)]))
        new_var = pd.DataFrame(index=list(self.adata.var_names) + new_variables)
        self.adata = sc.AnnData(
            X=new_X,
            var=new_var,
            obs=self.adata.obs,
            obsm=self.adata.obsm,
            uns=self.adata.uns)

    def ensure_not_view(self):
        """Ensures that the AnnData object saved within is not a view.
        """

        if self.adata.is_view:
            self.adata = self.adata.copy()

        else:
            pass

    def ensure_batch_assignment(self):
        """Ensures that self.batch_key is actually a key in self.adata.obs.
        """

        if self.batch_key not in self.adata.obs.columns:
            raise KeyError('The batch key supplied does not match any column in adata.obs.')

    def check_for_counts(self):
        """Checks self.adata.X and self.adata.raw.X for presence of raw count data.
        """

        if is_counts(self.adata.X):
            pass

        else:
            if hasattr(self.adata, 'raw') and self.adata.raw is not None and is_counts(self.adata.raw.X):
                self.adata.X = self.adata.raw.X

            elif hasattr(self.adata, 'layers') and 'raw' in self.adata.layers and is_counts(self.adata.layers['raw']):
                self.adata.X = self.adata.layers['raw']

            else:
                raise ValueError('No raw counts found in adata.X or adata.raw.X or adata.layers["raw"].')

    def ensure_normlog(self):
        if 'normlog' not in self.adata.layers:            
            copy_adata = self.adata.copy()
            sc.pp.normalize_total(copy_adata, target_sum=10000)
            sc.pp.log1p(copy_adata)
            self.adata.layers['normlog'] = copy_adata.X

    def throw_out_nan(self, adata, obs_key):
        adata = adata[adata.obs[obs_key] != '']
        adata = adata[adata.obs[obs_key] != 'nan']
        adata = adata[adata.obs[obs_key] != np.nan]
        adata = adata[(adata.obs[obs_key] != adata.obs[obs_key]) is not True]

        return adata

    def set_predictions(self, obs_key, barcodes, y_pred):
        """Add explanation.
        """

        barcodes_np_index = np.where(
            np.isin(
                np.array(self.adata.obs_names), 
                barcodes))[0]

        if f'{obs_key}_pred' in self.adata.obs.columns:
            self.adata.obs[f'{obs_key}_pred'] = pd.Categorical(self.adata.obs[f'{obs_key}_pred'])
            for cat in np.unique(y_pred):
                if cat not in self.adata.obs[f'{obs_key}_pred'].cat.categories:
                    self.adata.obs[f'{obs_key}_pred'].cat.add_categories(cat, inplace=True)

        try:
            existing_annotations = self.adata.obs[f'{obs_key}_pred']
            existing_annotations[barcodes_np_index] = y_pred
            self.adata.obs[f'{obs_key}_pred'] = existing_annotations

        except KeyError:            
            pred_template = np.empty(shape=len(self.adata))
            pred_template[:] = np.nan            
            # NEED TO TEST IF CONVERSION OF NP.NAN TO STR CREATES PROBLEMS
            pred_template = pred_template.astype(str)
            pred_template[barcodes_np_index] = y_pred
            self.adata.obs[f'{obs_key}_pred'] = pred_template

    def get_predicted_barcodes(self, obs_key, child_node, predicted_from=None):
        """Add explanation.
        """

        adata_subset = self.adata[self.adata.obs[f'{obs_key}_pred'] == child_node]
        if type(predicted_from) != type(None):
            predicted_from = list(predicted_from)
            predicted_barcodes = [b for b in adata_subset.obs_names if b in predicted_from]

        else:
            predicted_barcodes = adata_subset.obs_names

        return predicted_barcodes

    def get_total_accuracy(self, obs_key, test_barcodes):
        """Add explanation.
        """

        adata_subset = self.adata[test_barcodes, :]
        adata_subset = self.throw_out_nan(adata_subset, obs_key)
        known_type = np.array(adata_subset.obs[obs_key])
        pred_type = np.array(adata_subset.obs[f'{obs_key}_pred'])
        possible_labels = np.concatenate((known_type, pred_type))
        possible_labels = np.unique(possible_labels)
        acc = np.sum(known_type == pred_type, axis = 0) / len(known_type)
        con_mat = confusion_matrix(
            y_true=known_type, 
            y_pred=pred_type, 
            normalize='true',
            labels=possible_labels)
        acc = round(acc * 100, 2)
        print(f'Overall accuracy is {acc} %')
        disp = ConfusionMatrixDisplay(con_mat, display_labels=possible_labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')
        plt.show()

        return acc, con_mat, possible_labels

    def get_top_genes(self, classifier_node, barcodes, n_genes):
        if type(barcodes) == type(None):
            barcodes = self.adata.obs_names

        adata_subset = self.adata[barcodes, :].copy()
        sc.pp.normalize_total(adata_subset)
        sc.pp.log1p(adata_subset)
        obs_name_children = self.get_children_obs_key(classifier_node)
        sc.tl.rank_genes_groups(adata_subset, obs_name_children, n_genes=n_genes)
        total_top_genes = []
        for label in adata_subset.obs[obs_name_children].unique():
            for gene in list(adata_subset.uns['rank_genes_groups']['names'][label]):
                if gene not in total_top_genes:
                    total_top_genes.append(gene)

        return total_top_genes

    def feature_selection(self, barcodes_positive, barcodes_negative, data_type, n_features=300, method='f_classif', return_idx=False, max_n_features=10000):
        self.adata.obs.loc[barcodes_positive, 'node'] = 'yes'
        self.adata.obs.loc[barcodes_negative, 'node'] = 'no'
        barcodes = barcodes_positive + barcodes_negative        
        n_features = min(n_features, max_n_features)

        if method == 'hvg':
            if data_type == 'counts':
                flavor = 'seurat_v3'
                layer = None

            elif data_type == 'normlog':
                flavor = 'seurat'
                layer = 'normlog'

            else:
                raise Exception('HVG selection not implemented for embeddings.')

            table = sc.pp.highly_variable_genes(
                self.adata[barcodes, :], 
                n_top_genes=n_features, 
                inplace=False, 
                flavor=flavor,
                layer=layer)
            hv_genes = table[table['highly_variable']].index
            if type(hv_genes[0]) == int:
                hv_genes = list(self.adata.var.index[hv_genes])

            gc.collect()
            return hv_genes

        if data_type == 'normlog':
            x = self.adata[barcodes, :].layers['normlog']

        elif data_type == 'counts':
            x = self.adata[barcodes, :].X

        elif data_type in self.adata.obsm:
            x = self.adata[barcodes, :].obsm[data_type]

        else:
            raise Exception('Feature selection not implemeted for embeddings.')

        if hasattr(x, 'todense'):
            x = x.todense()
            x = np.asarray(x)

        x = z_transform_properties(x)
        y = self.adata[barcodes, :].obs['node'].values
        # Make sure the default n_features option does not lead to trying to select more features than available
        n_features = min(x.shape[1], n_features)
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        warnings.filterwarnings(action='ignore', category=UserWarning)

        

        selecter = SelectKBest(f_classif, k=n_features)
        selecter.fit(x, y)

        if return_idx:
            bool_idx = selecter.get_support()
            gc.collect()
            return list(np.where(bool_idx)[0])

        else:
            gc.collect()
            return list(self.adata.var_names[selecter.get_support()])

    def init_resampling(self, sampling_method, sampling_strategy='auto'):
        self.sampling_method = sampling_method
        self.sampling_strategy = sampling_strategy