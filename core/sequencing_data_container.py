import scvi
import os
import numpy as np
import scanpy as sc
import pandas as pd
from datetime import datetime
from classiFire.core.tools import is_counts
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy import sparse
import matplotlib.pyplot as plt

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
            self.adata = self.adata.copy()

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

        scvi_save_path = os.path.join(
            self.save_path, 
            'models', 
            'scvi')
        if not os.path.exists(scvi_save_path):
            os.makedirs(scvi_save_path)

        #### TODO: Robust model saving and loading
        models = [model for model in os.listdir(scvi_save_path) if model.endswith(key)]
        if len(models) > 0:
            model = models[-1]
            model_exists = True

        else:
            model = f'{self.scVI_model_prefix}_{key}'
            model_exists = False

        model_path = os.path.join(self.save_path,
            'models',
            'scvi', 
            model)        
        #model_exists = os.path.exists(model_path)
        if type(barcodes) != type(None):
            relevant_adata = self.adata[barcodes, :].copy()

        else:
            relevant_adata = self.adata

        scvi.model.SCVI.setup_anndata(
            relevant_adata,
            batch_key=self.batch_key)

        save_model = False
        if model_exists and not overwrite:
            vae = scvi.model.SCVI.load_query_data(
                relevant_adata,
                model_path)

        else:
            save_model = True
            arches_params = dict(
                use_layer_norm="both",
                use_batch_norm="none",
                encode_covariates=True,
                dropout_rate=0.2,
                n_layers=2,
                n_latent=n_dimensions)
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

    def ensure_normlog(self):
        if not 'normlog' in self.adata.layers:            
            copy_adata = self.adata.copy()
            sc.pp.normalize_total(copy_adata, target_sum=10000)
            sc.pp.log1p(copy_adata)
            self.adata.layers['normlog'] = copy_adata.X

    def get_x_y_untransformed(
        self, 
        barcodes, 
        obs_name_children, 
        data='normlog',
        var_names=None,
        scVI_key=None,
        return_adata=False):
        """Add explanation
        """
        if type(var_names) == type(None):
            var_names = list(self.adata.var_names)

        self.ensure_normlog()
        adata_subset = self.adata[barcodes, var_names]
        if data == 'normlog':
            x = adata_subset.layers['normlog']

        elif data == 'counts':
            x = adata_subset.X

        elif data == 'scVI':
            if type(scVI_key) == type(None):
                raise Exception('You are trying to use scVI data as training data, please supply a valid obsm key.')
                
            x = adata_subset.obsm[scVI_key]

        else:
            raise Exception('Please specify the type of data you want to supply to the classifier. Options are: "scVI", "counts" and "normlog".')
            
        if hasattr(x, 'todense'):
            x = x.todense()

        y = np.array(adata_subset.obs[obs_name_children])
        if hasattr(self, 'sampling_method') and type(self.sampling_method) != type(None):
            res = self.sampling_method(sampling_strategy=self.sampling_strategy)
            x_res, y_res = res.fit_resample(x, y)

        else:
            x_res, y_res = x, y

        if return_adata == False:
            return x_res, y_res

        elif data != 'scVI':
            adata_subset_res = sc.AnnData(x_res)
            adata_subset_res.var = pd.DataFrame(index=adata_subset.var_names)
            adata_subset_res.obs[obs_name_children] = y_res
            return adata_subset_res, obs_name_children

        else:
            raise Exception('Returning an AnnData object is not compatible with scVI data.')
    
    # def set_chi2_features(self, current_node, barcodes, children_obs_key):
    #     """
    #     Params: 
    #     -------------------------------------
    #     current_node: node of current local classifier 
    #     barcodes: current barcodes 
    #     children_obs_key: e.g. 'Level_4' (see list with hierarchical labels in init of hierarchical classifier)

    #     CAVE: 50 best features hard encoded

    #     """
        
    #     X = self.adata[barcodes].X 
    #     y = self.adata.obs[f'{children_obs_key}']

    #     X_chi2 = SelectKBest(chi2, k=50).fit_transform(X,y)

    #     return X_chi2

    #     # self.adata.obsm[f'X_chi2_{current_node}'] = X_chi2
    
    def get_x_untransformed(
        self, 
        barcodes, 
        data='normlog', 
        var_names=None, 
        scVI_key=None, 
        return_adata=False):
        """Add explanation.
        """
        
        if type(var_names) == type(None):
            var_names = self.adata.var_names

        # !!!! TODO: ERROR PREVENTION
        # Initialize unknown, but expected, vars as 0
        new_var_names = [v for v in var_names if not v in self.adata.var_names]
        if len(new_var_names) > 0:
            new_values = np.empty(
                (len(self.adata.obs_names), len(new_var_names))
            )
            new_values[:] = 0
            new_X = sparse.csr_matrix(
                sparse.hstack([self.adata.X, sparse.csr_matrix(new_values)]))
            new_var = pd.DataFrame(index=list(self.adata.var_names) + new_var_names)
            self.adata = sc.AnnData(
                X=sparse.csr_matrix(new_X), 
                var=new_var, 
                obs=self.adata.obs)
            self.ensure_normlog()

        adata_subset = self.adata[barcodes, var_names]
        if data == 'normlog':
            x = adata_subset.layers['normlog']

        elif data == 'counts':
            x = adata_subset.X

        elif data == 'scVI':
            if type(scVI_key) == type(None):
                raise Exception('You are trying to use scVI data as training data, please supply a valid obsm key.')

            x = adata_subset.obsm[scVI_key]

        else:
            raise Exception('Please specify the type of data you want to supply to the classifier. Options are: "scVI", "counts" and "normlog".')

        if hasattr(x, 'todense'):
            x = x.todense()

        if return_adata == False:
            return x

        elif data != 'scVI':
            adata_subset_selected = sc.AnnData(x)
            adata_subset_selected.var = pd.DataFrame(index=adata_subset.var_names)
            return adata_subset_selected

        else:
            raise Exception('Returning an AnnData object is not compatible with scVI data.')

    def get_true_barcodes(self, obs_name_node, node, true_from=None):
        """Retrieves bar codes of the cells that match the node supplied in the obs column supplied.
        Important for training using only the cells that are actually classified as belonging to
        that node.
        """

        adata_subset = self.adata[self.adata.obs[obs_name_node] == node]
        if type(true_from) != type(None):
            true_from = list(true_from)
            true_barcodes = [b for b in adata_subset.obs_names if b in true_from]

        else:
            true_barcodes = adata_subset.obs_names

        return true_barcodes

    def set_predictions(self, obs_key, barcodes, y_pred):
        """Add explanation.
        """

        barcodes_np_index = np.where(
            np.isin(
                np.array(self.adata.obs_names), 
                barcodes))[0]
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

    def set_prob_based_predictions(self, node, children_obs_key, parent_obs_key, barcodes, y_pred, fitted_label_encoder):
        """Set predicitons where confidence is high enough
            REMEMBER currently only implemented with NeuralNetwork (y_pred is integer_vector! 
            has to be inverse transformed after nans are eliminated)
        """

        print(f'children_obs_key: {children_obs_key}')
        print(f'parent_obs_key: {parent_obs_key}')


        
        # #with indices
        # current_barcode_subset_idx = np.where(
        #         np.isin(np.array(self.adata.obs_names), 
        #                 barcodes))[0]
        # #real barcode names to avoid conflicts with indices when assigning predicted and non predicted cells
        # current_barcode_subset = np.array(self.adata.obs_names)[current_barcode_subset_idx]

        # print(f'current_barcode_subset (should be obs names): {current_barcode_subset}')
        
        #check where predictions were made and where not
        barcodes_with_prediction_idx = np.argwhere(~np.isnan(y_pred)).flatten()
        barcodes_no_prediction_idx = np.argwhere(np.isnan(y_pred)).flatten()

        #row names instead of indices
        barcodes_with_prediction = np.array(barcodes)[barcodes_with_prediction_idx]
        barcodes_no_prediction = np.array(barcodes)[barcodes_no_prediction_idx]


        print(f'consistency_check: len(with_pred) + len(no_pred): {len(barcodes_with_prediction_idx) + len(barcodes_no_prediction_idx)} sould = len(y_pred): {len(y_pred)}\n')
        print(f'and should be equal to len(current_barcode_subset): {len(barcodes)}')
        
        
        # print(barcodes_with_prediction_idx[:15])
        # print(y_pred[:15])
        # print(np.array(y_pred)[barcodes_with_prediction_idx][:15])
        # print(fitted_label_encoder.inverse_transform(np.array(y_pred)[barcodes_with_prediction_idx].astype(int)))

        #make y_pred contain string labels of cells (only where entry != nan)
        y_pred_labels = fitted_label_encoder.inverse_transform(np.array(y_pred)[barcodes_with_prediction_idx].astype(int))


        #write predictions/non preds in Andnata, TODO SAVE FINAL PRED LEVEL IN ANOTHER OBS

        # 1.) set values where predictions were made
        try: 
            print(f'adata.obs."children_obs_key_pred" value counts vor setzen der y_preds mit prediction aus "try":') 
            print(self.adata.obs[f'{children_obs_key}_pred'][barcodes].value_counts())
            existing_annotations = self.adata.obs[f'{children_obs_key}_pred']
            existing_annotations.loc[barcodes_with_prediction] = y_pred_labels
            print(f'adata.obs."children_obs_key_pred" value counts nach setzen der y_preds mit prediction aus "try":') 
            print(self.adata.obs[f'{children_obs_key}_pred'][barcodes].value_counts())
        except KeyError:
            print(f'went to key error with prediction in node {node}')
            pred_template = np.empty(shape=len(self.adata))
            pred_template[:] = np.nan            
            # NEED TO TEST IF CONVERSION OF NP.NAN TO STR CREATES PROBLEMS
            pred_template = pred_template.astype(str)

            pred_template[barcodes_with_prediction_idx] = y_pred_labels
            self.adata.obs[f'{children_obs_key}_pred'] = pred_template
            print(f'adata.obs."children_obs_key_pred" value counts nach setzen der y_preds mit prediction aus "except":') 
            print(self.adata.obs[f'{children_obs_key}_pred'][barcodes].value_counts())
            

        # 2.) set values where no prediction was made to current label (node of current classifier)
        #SHOULDN'T BE NECESSARY; cells are already subsetted to this node, thus have this annotation already,
        #excpet for the very first used level
        try: 
            print(f'adata.obs."parent_obs_key_pred" value counts vor setzen der y_preds ohne prediction aus "try":') 
            print(self.adata.obs[f'{parent_obs_key}_pred'][barcodes].value_counts())
            existing_annotations = self.adata.obs[f'{parent_obs_key}_pred']
            existing_annotations.loc[barcodes_no_prediction] = f'{node}'
            print(f'adata.obs."parent_obs_key_pred" value counts nach setzen der y_preds ohne prediction aus "try":') 
            print(self.adata.obs[f'{parent_obs_key}_pred'][barcodes].value_counts())
        except KeyError:
            print(f'went to key error no prediction in node {node}')

            existing_annotations_template = np.empty(shape=len(self.adata))
            existing_annotations_template[:] = np.nan
            existing_annotations_template = existing_annotations_template.astype(str)
            
            existing_annotations_template[barcodes_no_prediction_idx] = f'{node}'
            self.adata.obs[f'{parent_obs_key}_pred'] = existing_annotations_template
            print(f'adata.obs."parent_obs_key_pred" value counts nach setzen der y_preds ohne prediction aus "except":') 
            print(self.adata.obs[f'{parent_obs_key}_pred'][barcodes].value_counts())


        #write where the last annotation level for a cell is currently saved (i.e. write the current obs_key for a cell 
        #in 'final_pred_obs_key' - will be the last for that cell when not overwritten by next level classifier)
        try: 
            print(f'adata.obs."final_pred_obs_key" value counts vor setzen der final_preds prediction aus "try":') 
            print(self.adata.obs[f'final_pred_obs_key'][barcodes].value_counts())
            existing_final_pred_vec = self.adata.obs[f'final_pred_obs_key']
            #use real cell label names instead of indices to avoid working on subsets
            existing_final_pred_vec.loc[barcodes_with_prediction] = f'{children_obs_key}'
            existing_final_pred_vec.loc[barcodes_no_prediction] = f'{parent_obs_key}'
            self.adata.obs[f'final_pred_obs_key'] = existing_final_pred_vec
            print(f'adata.obs."final_pred_obs_key" value counts nach setzen der final_preds prediction aus "try":') 
            print(self.adata.obs[f'final_pred_obs_key'][barcodes].value_counts())

        except KeyError:
            print(f'went to key error final prediction obs_key in node {node}')
            final_pred_template = np.empty(shape=len(self.adata))
            final_pred_template[:] = np.nan            
            final_pred_template = final_pred_template.astype(str)
            final_pred_template[barcodes_with_prediction_idx] = f'{children_obs_key}'
            try:
                final_pred_template[barcodes_no_prediction_idx] = f'{parent_obs_key}' 
            except:
                #exception in case parent_obs_key not defined, should only happen for 
                #first classifier, in that case should really set first node in hierarchy
                final_pred_template[barcodes_no_prediction_idx] = 'funny things happening'

            self.adata.obs[f'final_pred_obs_key'] = final_pred_template
            print(f'adata.obs."final_pred_obs_key" value counts nach setzen der final_preds prediction aus "try":') 
            print(self.adata.obs[f'final_pred_obs_key'][barcodes].value_counts())


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
        disp.plot(xticks_rotation='vertical')
        plt.show()

        return acc, con_mat, possible_labels

    def get_hierarchical_accuracy(self, test_barcodes, level_obs_keys, all_labels, overview_obs_key = None):
        '''Calculate conmats and accuracy with respect to the actually reached levels 
            in the non- obligatory leaf hierachical classifier
            
            Params:
            -------------------------------------------------------------------------
            test_barcodes: only calculate results for cells that were used for testing
            
            level_obs_keys: (array-like) obs_keys for the levels of the hierarchy, specified in 
            hierarchy container constructor

            all_labels: array-like of all labels in the hierarchy 
            (should be called with Hierarchical_classifier.hierachry_container.all_nodes)

            overview_obs_key: sets the level for which finest labels are supposed to be compared with
            '''
        adata_subset = self.adata[test_barcodes, :]
        final_obs_key = level_obs_keys[-1]
        # adata_final_subset = adata_subset[adata_subset.obs.final_pred_obs_key == f'{final_obs_key}', :]
        known_type = adata_subset.obs[final_obs_key]
        #calculate accuray matrix only for those cells that really did reach the final level 
    
        # known_type = np.array(adata_final_subset.obs[final_obs_key])

        final_pred_levels = adata_subset.obs['final_pred_obs_key']
        pred_type = []
        for obs_name, final_pred_level in zip(adata_subset.obs_names,final_pred_levels):
            pred_type.append(adata_subset.obs[f'{final_pred_level}_pred'].loc[obs_name])

        # pred_type = np.array(adata_final_subset.obs[f'{final_obs_key}_pred'])
        possible_labels = np.concatenate((known_type, pred_type))
        possible_labels = np.unique(possible_labels)
        acc = np.sum(known_type == pred_type, axis = 0) / len(known_type)
        try:
            con_mat = confusion_matrix(
                y_true=known_type, 
                y_pred=pred_type, 
                normalize='true',
                labels=possible_labels)
            acc = round(acc * 100, 2)
            print(f'Overall accuracy is {acc} %')
            disp = ConfusionMatrixDisplay(con_mat, display_labels=possible_labels)
            disp.plot()
        except:
            print('something didnt work')
            return

        if overview_obs_key != None:

            print('in overview_obs part')

            adata_subset = self.adata[test_barcodes, :]
            final_obs_key = overview_obs_key
            # adata_final_subset = adata_subset[adata_subset.obs.final_pred_obs_key == f'{final_obs_key}', :]
            known_type = adata_subset.obs[final_obs_key]
            #calculate accuray matrix only for those cells that really did reach the final level 
    
            # known_type = np.array(adata_final_subset.obs[final_obs_key])

            final_pred_levels = adata_subset.obs['final_pred_obs_key']
            pred_type = []
            for obs_name, final_pred_level in zip(adata_subset.obs_names,final_pred_levels):
                pred_type.append(adata_subset.obs[f'{final_pred_level}_pred'].loc[obs_name])

            # pred_type = np.array(adata_final_subset.obs[f'{final_obs_key}_pred'])
            possible_labels = np.concatenate((known_type, pred_type))
            possible_labels = np.unique(possible_labels)
            acc = np.sum(known_type == pred_type, axis = 0) / len(known_type)
            try:
                con_mat = confusion_matrix(
                    y_true=known_type, 
                    y_pred=pred_type, 
                    normalize='true',
                    labels=possible_labels)
                acc = round(acc * 100, 2)
                print(f'Overall accuracy is {acc} %')
                disp = ConfusionMatrixDisplay(con_mat, display_labels=possible_labels)
                disp.plot()
            except:
                print('something didnt work')
                return



        # if overview_obs_key != None:

        #     y_known = adata_subset.obs[f'{overview_obs_key}']
        #     y_pred = []
        #     #vector with all final decisions (could be quite slow for large datasets)
        #     for barcode in test_barcodes:
        #         final_pred_level = adata_final_subset[barcode].obs.final_pred_obs_key
        #         final_decision = adata_final_subset[barcode].obs[f'{final_pred_level}_pred']
        #         y_pred.append(final_decision)

        #     possible_labels_overview = np.concatenate((known_type, pred_type))
        #     possible_labels_overview = np.unique(possible_labels)
        #     try:
        #         con_mat_overview = confusion_matrix(
        #             y_true=known_type, 
        #             y_pred=pred_type, 
        #             normalize='true',
        #             labels=possible_labels)
        #         disp = ConfusionMatrixDisplay(con_mat_overview, display_labels=possible_labels_overview)
        #         disp.plot()
        #         return acc, con_mat, possible_labels, con_mat_overview, possible_labels_overview
        #     except:
        #         'No plot for now'
            
            

        else:
            return acc, con_mat, possible_labels
       
    def get_top_genes(self, barcodes, obs_name_children, n_genes):
        if type(barcodes) == type(None):
            barcodes = self.adata.obs_names

        adata_subset = self.adata[barcodes, :].copy()
        sc.pp.normalize_total(adata_subset)
        sc.pp.log1p(adata_subset)
        sc.tl.rank_genes_groups(adata_subset, obs_name_children, n_genes=n_genes)
        total_top_genes = []
        for label in adata_subset.obs[obs_name_children].unique():
            for gene in list(adata_subset.uns['rank_genes_groups']['names'][label]):
                if not gene in total_top_genes:
                    total_top_genes.append(gene)

        return total_top_genes

    def init_resampling(self, sampling_method, sampling_strategy='auto'):
        self.sampling_method = sampling_method
        self.sampling_strategy = sampling_strategy

    def load_new_adata(self, adata):
        self.adata = adata