import gc
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.linear_model import LogisticRegression
from Compocyte.core.tools import is_counts, z_transform_properties
from scipy import sparse
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

class DataBase():
    """Add explanation
    """

    def load_adata(
        self,
        adata):

        self.adata = adata
        if self.var_names is not None: # Load new adata for transfer learning/prediction
            self.adata = self.variable_match_adata(adata)

        else:
            self.var_names = self.adata.var_names.tolist()

        self.ensure_not_view()
        self.check_for_counts()
        if self.default_input_data == 'normlog':
            self.ensure_normlog()

        if hasattr(self.adata.X, 'todense'):
            self.adata.X = np.array(self.adata.X.todense())        

    def variable_match_adata(
        self, 
        new_adata):

        new_var_names = [v for v in self.var_names if v not in new_adata.var_names]
        if is_counts(new_adata.X):
            pass

        else:
            if hasattr(new_adata, 'raw') and new_adata.raw is not None and is_counts(new_adata.raw.X):
                new_adata.X = new_adata.raw[new_adata.obs_names, new_adata.var_names].X #new_adata.raw.X

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
                obs=new_adata.obs)

        return new_adata[:, self.var_names]

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

    def check_for_counts(self):
        """Checks self.adata.X and self.adata.raw.X for presence of raw count data.
        """

        if self.ignore_counts:
            return

        if is_counts(self.adata.X):
            pass

        else:
            if hasattr(self.adata, 'raw') and self.adata.raw is not None and is_counts(self.adata.raw.X):
                self.adata.X = self.adata.raw[self.adata.obs_names, self.adata.var_names].X #self.adata.raw.X

            elif hasattr(self.adata, 'layers') and 'raw' in self.adata.layers and is_counts(self.adata.layers['raw']):
                self.adata.X = self.adata.layers['raw']

            else:
                raise ValueError('No raw counts found in adata.X or adata.raw.X or adata.layers["raw"].')

    def ensure_normlog(self):
        is_normlog = np.all(
            np.round(
                np.sum(np.expm1(self.adata.X), axis=1)
            ) == 10000)
        if not is_normlog:            
            sc.pp.normalize_total(self.adata, target_sum=10000)
            sc.pp.log1p(self.adata)

    def set_predictions(self, obs_key, barcodes, y_pred):
        """Add explanation.
        """

        barcodes_np_index = np.where(
            np.isin(
                np.array(self.adata.obs_names), 
                barcodes))[0]

        if f'{obs_key}_pred' in self.adata.obs.columns:
            self.adata.obs[f'{obs_key}_pred'] = self.adata.obs[f'{obs_key}_pred'].astype(str)

        try:
            existing_annotations = self.adata.obs[f'{obs_key}_pred']
            existing_annotations[barcodes_np_index] = y_pred
            self.adata.obs[f'{obs_key}_pred'] = existing_annotations
            self.adata.obs[f'{obs_key}_pred'] = pd.Categorical(self.adata.obs[f'{obs_key}_pred'])

        except KeyError:            
            pred_template = np.empty(shape=len(self.adata))
            pred_template[:] = np.nan            
            # NEED TO TEST IF CONVERSION OF NP.NAN TO STR CREATES PROBLEMS
            pred_template = pred_template.astype(str)
            pred_template[barcodes_np_index] = y_pred
            self.adata.obs[f'{obs_key}_pred'] = pred_template

    def get_predicted_barcodes(self, obs_key, child_node):
        """Add explanation.
        """

        adata_subset = self.adata[self.adata.obs[f'{obs_key}_pred'] == child_node]

        return list(adata_subset.obs_names)

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
    
    def feature_selection_LCPN(self, relevant_cells, children_obs_key, n_features, return_idx=False):
        y = np.array(relevant_cells.obs[children_obs_key])
        x = np.array(relevant_cells.X)
        x = z_transform_properties(x)        
        if self.feature_select_using_LR:
            if len(x) > 50_000:
                choice = np.random.random_integers(0, len(x) - 1, 50_000)
                y = y[choice]
                x = x[choice, :]

            classifier = LogisticRegression(C=1.0, solver='sag', max_iter=500, multi_class='auto', class_weight='balanced')
            classifier.fit(x, y)
            gene_index = np.argpartition(np.abs(classifier.coef_), -n_features, axis = 1)[:, -n_features:]
            gene_index = np.unique(gene_index)
            genes = self.adata.var_names[gene_index]

            return list(genes)
        
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        warnings.filterwarnings(action='ignore', category=UserWarning)
        # Make sure the default n_features option does not lead to trying to select more features than available
        n_features = min(x.shape[1], n_features)
        selecter = SelectKBest(f_classif, k=n_features)
        selecter.fit(x, y)

        if return_idx:
            bool_idx = selecter.get_support()
            gc.collect()
            return list(np.where(bool_idx)[0])

        else:
            gc.collect()
            return list(self.adata.var_names[selecter.get_support()])