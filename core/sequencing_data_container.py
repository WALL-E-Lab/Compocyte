import scvi
import os
import numpy as np
from datetime import datetime
from classiFire.core.tools import is_counts

class SequencingDataContainer():
    """Add explanation
    """

    def __init__(self, adata, save_path, batch_key='batch', scVI_model_prefix=None):
        self.adata = adata
        self.save_path = save_path
        self.batch_key = batch_key
        self.scVI_model_prefix = scVI_model_prefix
        self.ensure_not_view()
        self.check_for_counts()
        self.ensure_batch_assignment()

    def ensure_not_view(self):
        """Ensures that the AnnData object saved within is not a view.
        """

        if self.adata.is_view:
            self.adata = adata.copy()

        else:
            pass

    def ensure_batch_assignment(self):
        """Ensures that self.batch_key is actually a key in self.adata.obs.
        """

        if not self.batch_key in self.adata.obs.columns:
            raise KeyError('The batch key supplied does not match any column in adata.obs.')

    def check_for_counts(self):
        """Checks self.adata.X and self.adata.raw.X for presence of raw count data.
        """

        if is_counts(self.adata.X):
            pass

        else:
            if hasattr(self.adata, 'raw') and self.adata.raw != None and is_counts(self.adata.raw.X):
                pass

            else:
                raise ValueError('No raw counts found in adata.X or adata.raw.X.')

    def get_scVI_key(
        self,
        node,
        n_dimensions=10,
        barcodes=None, 
        overwrite=False, 
        **kwargs):
        """This method ensures that scVI data is present as requested, i. e. for the specified
        barcodes, the specified node and the specified number of dimensions. Returns the obsm key
        corresponding to the requested scVI data.

        Parameters
        ----------
        node
            Node in the cell label hierarchy for which the scVI data should be specific.
        n_dimensions
            Number of latent dimensions/nodes in the scVI bottleneck.
        barcodes
            Barcodes belonging to cells which are to be predicted/trained with and for whom,
            consequently, scVI dimensions should be available.
        overwrite
            Whether or not existing scVI dimensions fitting all criteria should be overwritten.
        """
        
        key = f'X_scVI_{n_dimensions}_{node}'
        # Run scVI if it has not been run at the specified number of dimensions or for the specified
        # node
        if not key in self.adata.obsm or overwrite:
            self.run_scVI(
                n_dimensions, 
                key, 
                barcodes=barcodes, 
                overwrite=overwrite, 
                **kwargs)

        # Also run scVI again if entry does not exist for all barcodes supplied
        elif type(barcodes) != type(None):
            adata_subset = self.adata[barcodes, :]
            if np.isnan(adata_subset.obsm[key]).any():
                self.run_scVI(
                    n_dimensions, 
                    key, 
                    barcodes=barcodes, 
                    overwrite=overwrite, 
                    **kwargs)

        return key

    def run_scVI(
        self, 
        n_dimensions, 
        key, 
        barcodes=None, 
        overwrite=False, 
        **kwargs):
        """Run scVI with parameters taken from the scvi-tools scANVI tutorial.

        Parameters
        ----------
        n_dimensions
            Number of latent dimensions/nodes in the scVI bottleneck.
        key
            Key under which to save the latent representation in self.adata.obsm.
        barcodes
            Barcodes belonging to cells which are to be predicted/trained with and for whom,
            consequently, scVI dimensions should be available.
        overwrite
            Whether or not existing scVI dimensions fitting all criteria should be overwritten.
        """

        # Check if scVI has previously been trained for this node and number of dimensions
        model_path = os.path.join(
            self.save_path, 
            'models', 
            'scvi', 
            f'{self.scVI_model_prefix}_{key}')        
        model_exists = os.path.exists(model_path)
        if type(barcodes) != type(None):
            relevant_adata = self.adata[barcodes, :].copy()

        else:
            relevant_adata = self.adata

        scvi.model.SCVI.setup_anndata(
            relevant_adata, 
            batch_key=self.batch_key)

        save_model = False
        if model_exists and not overwrite:
            vae = scvi.model.SCVI.load(
                model_path,
                relevant_adata)

        else:
            save_model = True
            arches_params = dict(
                use_layer_norm="both",
                use_batch_norm="none",
                encode_covariates=True,
                dropout_rate=0.2,
                n_layers=2,)
            vae = scvi.model.SCVI(
                relevant_adata,
                **arches_params)

        vae.train(
            early_stopping=True,
            early_stopping_patience=10)
        if save_model:
            self.scVI_model_prefix = datetime.now().isoformat(timespec='minutes')
            model_path = os.path.join(
                self.save_path, 
                'models', 
                'scvi', 
                f'{self.scVI_model_prefix}_{key}')
            vae.save(
                model_path,
                overwrite=True)

        if type(barcodes) != type(None):
            # Ensure that scVI values in the relevant obsm key are only being set for those
            # cells that belong to the specified subset (by barcodes) and values for all other
            # cells are set to np.nan
            scvi_template = np.empty(shape=(len(self.adata), n_dimensions))
            scvi_template[:] = np.nan
            barcodes_np_index = np.where(
                np.isin(
                    np.array(self.adata.obs_names), 
                    barcodes))[0]
            scvi_template[barcodes_np_index, :] = vae.get_latent_representation()
            self.adata.obsm[key] = scvi_template

        else:
            self.adata.obsm[key] = vae.get_latent_representation()

    def get_x_y_untransformed(self, barcodes, scVI_key, obs_name_children):
        """Add explanation.
        """

        adata_subset = self.adata[barcodes, :]
        x = adata_subset.obsm[scVI_key]
        y = np.array(adata_subset.obs[obs_name_children])

        return x, y

    def get_x_untransformed(self, barcodes, scVI_key):
        """Add explanation.
        """

        adata_subset = self.adata[barcodes, :]
        x = adata_subset.obsm[scVI_key]

        return x

    def get_true_barcodes(self, obs_name_node, node, true_from=None):
        """Retrieves bar codes of the cells that match the node supplied in the obs column supplied.
        Important for training using only the cells that are actually classified as belonging to
        that node.
        """

        adata_subset = self.adata[self.adata.obs[obs_name_node] == node]
        if type(true_from) != type(None):
            true_barcodes = [b for b in adata_subset.obs_names if b in true_from]

        else:
            true_barcodes = adata_subset.obs_names

        return true_barcodes

    def set_predictions(self, node, barcodes, y_pred):
        """Add explanation.
        """

        pred_template = np.empty(shape=len(self.adata))
        pred_template[:] = np.nan
        barcodes_np_index = np.where(
            np.isin(
                np.array(self.adata.obs_names), 
                barcodes))[0]
        # NEED TO TEST IF CONVERSION OF NP.NAN TO STR CREATES PROBLEMS
        pred_template = pred_template.astype(str)
        pred_template[barcodes_np_index] = y_pred
        self.adata.obs[f'{node}_pred'] = pred_template