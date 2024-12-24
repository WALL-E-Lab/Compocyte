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