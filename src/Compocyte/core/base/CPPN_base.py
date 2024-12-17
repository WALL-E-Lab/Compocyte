from Compocyte.core.tools import z_transform_properties
from Compocyte.core.models.dense import DenseKeras
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.log_reg import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from Compocyte.core.tools import flatten_dict, make_graph_from_edges, set_node_to_depth
from uncertainties import ufloat
from copy import deepcopy
from time import time
import tensorflow.keras as keras
import numpy as np
import networkx as nx
import gc
import multiprocessing as mp

class CPPNBase():

    def train_single_node_CPPN_ensemble(
        self,
        node, 
        n_ensemble_networks:int,
        train_barcodes = None):
        """
        Initialize training of a classifier ensemble at `node`.
        Follows preprint https://arxiv.org/abs/1612.01474.

        Parameters
        ----------
        node 
            Node in hierarchical graph which is supposed to be trained. 
        n_ensemble_networks `int`
            Number of ensembles to be trained.
        train_barcodes 
            Barcodes of cells used for training.
        """


        self.graph.nodes[node]["trained_ensemble"] = dict()
        # {
        #         "ensemble_classifier": [trained_networks[i].get("local_classifier") for i in range(len(trained_networks))]
        #         }

        

        #copy train_single_node_CPPN code for ensemble situation

        if train_barcodes is None:
            train_barcodes = self.adata.obs_names

        type_classifier = self.get_preferred_classifier(node)
        if type_classifier is None:
            type_classifier = DenseTorch

        if self.default_input_data in type_classifier.possible_data_types or self.default_input_data in self.adata.obsm:
            data_type = self.default_input_data

        else:
            data_type = type_classifier.possible_data_types[0]
            print(f'{self.default_input_data} data is not currently compatible with {type_classifier}. Set to {data_type}')

        parent_obs_key = self.get_parent_obs_key(node)
        children_obs_key = self.get_children_obs_key(node)
        child_nodes = self.get_child_nodes(node)

        # Avoid problems with argmax for prediction by ensuring output activation is 2d
        output_len = max(len(child_nodes), 2)
        is_parent_node = self.adata.obs[parent_obs_key] == node
        is_child_node = self.adata.obs[children_obs_key].isin(child_nodes)
        potential_cells = self.adata[is_parent_node & is_child_node]
        # To avoid training with cells that should be reserved for testing, make sure to limit
        # to train_barcodes.
        relevant_cells = self.adata[[b for b in potential_cells.obs_names if b in train_barcodes], :]
        if data_type == 'normlog':
            self.ensure_normlog()

        n_cell_types = len(relevant_cells.obs[children_obs_key].unique())
        if n_cell_types == 0:
            return

        if data_type in ['counts', 'normlog']:
            selected_var_names = list(self.adata.var_names) 
        elif data_type in self.adata.obsm:
            selected_var_names = list(range(self.adata.obsm[data_type].shape[1]))
        else:
            raise Exception('Data type not currently supported.')
        
        if 'selected_var_names' in self.graph.nodes[node].keys():
            selected_var_names = self.graph.nodes[node]['selected_var_names']

        # Cannot define relevant genes for prediction if there is no cell group to compare to
        elif self.use_feature_selection and n_cell_types > 1:
            # Reinitialize classifier if, for the first time, more than one cell type is present
            if 'local_classifier' in self.graph.nodes[node]:
                del self.graph.nodes[node]['local_classifier']
            if 'trained_ensemble' in self.graph.nodes[node]:
                del self.graph.nodes[node]['trained_ensemble']

            return_idx = data_type not in ['counts', 'normlog']
            selected_var_names = []
            pct_relevant = len(relevant_cells) / len(self.adata)
            # n_features is simply overwritten if method=='hvg'
            projected_relevant_cells = pct_relevant * self.projected_total_cells
            # should not exceed a ratio of 1:100 of features to training samples as per google rules of ml
            n_features_by_samples = int(projected_relevant_cells / 100)
            n_features = max(
                min(self.max_features, n_features_by_samples),
                self.min_features
            )
            selected_var_names = self.feature_selection_CPPN(
                relevant_cells, 
                children_obs_key, 
                data_type, 
                n_features, 
                return_idx=return_idx)
    
        print('Selected genes first defined.')
        self.graph.nodes[node]['selected_var_names'] = selected_var_names
        n_input = len(selected_var_names)
        # TODO: Find permanent solution for training when only one child label is available
        # 1) It makes no sense to train a classifier to discriminate between multiple labels
        # when only one is available
        # 2) At the same time, compatibility with the usual training process must be maintained
        # 3) Stopping all cells at this level would unnecessarily reduce performance
        # 4) implement a way to classify yes/no? (compare against what exactly?)
        sequential_kwargs = self.sequential_kwargs
        if n_cell_types == 1:
            hidden_layers = [64, 10]


        #NOTE swapped with ensure_existence_classifier
        if data_type == 'counts':
            x = relevant_cells[:, selected_var_names].X
        elif data_type == 'normlog':
            x = relevant_cells[:, selected_var_names].layers['normlog']
        elif data_type in relevant_cells.obsm:
            x = relevant_cells.obsm[data_type][:, selected_var_names]
        else:
            raise Exception('Data type not currently supported.')
        
        if hasattr(x, 'todense'):
            x = x.todense()

        y = np.array(relevant_cells.obs[children_obs_key])
        if hasattr(self, 'sampling_method') and type(self.sampling_method) != type(None):
            res = self.sampling_method(sampling_strategy=self.sampling_strategy)
            x, y = res.fit_resample(x, y)

        if "label_encoding" not in self.graph.nodes[node]:
            self.graph.nodes[node]['label_encoding'] = {}

        for label in child_nodes:
            if label in self.graph.nodes[node]['label_encoding']:
                continue

            idx = len(self.graph.nodes[node]['label_encoding'])
            self.graph.nodes[node]['label_encoding'][label] = idx


        y_int = np.array(
            [self.graph.nodes[node]['label_encoding'][label] for label in y]
        ).astype(int)
        y_onehot = keras.utils.to_categorical(y_int, num_classes=output_len)
        x = z_transform_properties(x)

        #establish key in node
        self.graph.nodes[node]['trained_ensemble'] = []

        for network_i in range(n_ensemble_networks):
            #TODO: parallelize as well
            #needs randomization of initial parameters! (xavier.uniform already random)
            #not necessarily: adversarial training, different train data (might be disadvantageous for NN's)
            # trained_network_params = self.train_single_node_CPPN(node, train_barcodes) 
            # trained_networks.append(trained_network_params)
            
            #=======================
            # self.ensure_existence_classifier(
            # node, 
            # n_input,
            # n_output=output_len,
            # classifier=type_classifier,
            # sequential_kwargs=sequential_kwargs)

            # following code replaces ensure_existence_classifier call

            print(f'Training {network_i}. classifier at {node}.')

            output_len = len(list(self.graph.adj[node].keys()))

            if 'preferred_classifier' in self.graph.nodes[node].keys():
                self.graph.nodes[node]['trained_ensemble'].append(self.graph.nodes[node]['preferred_classifier'](n_input=n_input, n_output=output_len, **sequential_kwargs))
            else:
                self.graph.nodes[node]['trained_ensemble'].append(DenseTorch(n_input=n_input, n_output=output_len, **sequential_kwargs))

            try:
                #after previous if/else no indexError should occur in the following try block
                if hasattr(self, 'default_input_data') and self.default_input_data in self.graph.nodes[node]['trained_ensemble'][network_i].possible_data_types or self.default_input_data in self.adata.obsm:
                    self.graph.nodes[node]['trained_ensemble'][network_i].set_data_type(self.default_input_data)
            except AttributeError:
                pass # occurs whent trying to set data type for imported log reg model

            print(f"Data type for {node} set to {self.graph.nodes[node]['trained_ensemble'][network_i].data_type}")

            #======================

            
            self.graph.nodes[node]['trained_ensemble'][network_i]._train(x=x, y_onehot=y_onehot, y=y, y_int=y_int, train_kwargs=self.train_kwargs)
            timestamp = str(time()).replace('.', '_')
            
            if node not in self.trainings.keys():
                self.trainings[node] = {}

            self.trainings[node][timestamp] = {
                'barcodes': list(relevant_cells.obs_names),
                'node': node,
                'data_type': data_type,
                'type_classifier': type(self.graph.nodes[node]['local_classifier']),
                'var_names': selected_var_names
            }
            gc.collect()



        # if n_ensemble_networks == len(trained_networks):
        #     #store ensemble classifier in list as new node attribute
        #     #label encoder and selected_var_names are taken from default node attributes (overwritten each time 
        #     # #self.train_single_node_CPPN is called - identical behaviour?)
        #     self.graph.nodes[node]["trained_ensemble"] = {
        #         "ensemble_classifier": [trained_networks[i].get("local_classifier") for i in range(len(trained_networks))]
        #         }


    def train_single_node_CPPN(
        self, 
        node, 
        train_barcodes=None,
        ensemble_learning=False,
        n_ensemble_networks=5):
        """Trains the local classifier stored at node.

        Parameters
        ----------
        node
            Specifies which local classifier is to be trained. node='T' would result in training
            of the classifier further differentiating between T cells.
        barcodes
            Specifies which cells should be used for training.
        ensemble_learning 
            If `True`, an ensemble is trained at each node. TODO: at each specified node or exclude nodes
        """

        if ensemble_learning: 
            return self.train_single_node_CPPN_ensemble(node, 
                                                        train_barcodes=train_barcodes, 
                                                        n_ensemble_networks=n_ensemble_networks)

        if train_barcodes is None:
            train_barcodes = self.adata.obs_names

        type_classifier = self.get_preferred_classifier(node)
        if type_classifier is None:
            type_classifier = DenseTorch

        if self.default_input_data in type_classifier.possible_data_types or self.default_input_data in self.adata.obsm:
            data_type = self.default_input_data

        else:
            data_type = type_classifier.possible_data_types[0]
            print(f'{self.default_input_data} data is not currently compatible with {type_classifier}. Set to {data_type}')

        parent_obs_key = self.get_parent_obs_key(node)
        children_obs_key = self.get_children_obs_key(node)
        child_nodes = self.get_child_nodes(node)
        # Avoid problems with argmax for prediction by ensuring output activation is 2d
        output_len = max(len(child_nodes), 2)
        is_parent_node = self.adata.obs[parent_obs_key] == node
        is_child_node = self.adata.obs[children_obs_key].isin(child_nodes)
        potential_cells = self.adata[is_parent_node & is_child_node]
        # To avoid training with cells that should be reserved for testing, make sure to limit
        # to train_barcodes.
        relevant_cells = self.adata[[b for b in potential_cells.obs_names if b in train_barcodes], :]
        if data_type == 'normlog':
            self.ensure_normlog()

        n_cell_types = len(relevant_cells.obs[children_obs_key].unique())
        if n_cell_types == 0:
            return

        print(f'Training at {node}.')
        if data_type in ['counts', 'normlog']:
            selected_var_names = list(self.adata.var_names) 

        elif data_type in self.adata.obsm:
            selected_var_names = list(range(self.adata.obsm[data_type].shape[1]))

        else:
            raise Exception('Data type not currently supported.')

        if 'selected_var_names' in self.graph.nodes[node].keys():
            selected_var_names = self.graph.nodes[node]['selected_var_names']

        # Cannot define relevant genes for prediction if there is no cell group to compare to
        elif self.use_feature_selection and n_cell_types > 1:
            # Reinitialize classifier if, for the first time, more than one cell type is present
            if 'local_classifier' in self.graph.nodes[node]:
                del self.graph.nodes[node]['local_classifier']

            return_idx = data_type not in ['counts', 'normlog']
            selected_var_names = []
            pct_relevant = len(relevant_cells) / len(self.adata)
            # n_features is simply overwritten if method=='hvg'
            projected_relevant_cells = pct_relevant * self.projected_total_cells
            # should not exceed a ratio of 1:100 of features to training samples as per google rules of ml
            n_features_by_samples = int(projected_relevant_cells / 100)
            n_features = max(
                min(self.max_features, n_features_by_samples),
                self.min_features
            )
            selected_var_names = self.feature_selection_CPPN(
                relevant_cells, 
                children_obs_key, 
                data_type, 
                n_features, 
                return_idx=return_idx)

            print('Selected genes first defined.')
            self.graph.nodes[node]['selected_var_names'] = selected_var_names  

        n_input = len(selected_var_names)        
        self.ensure_existence_classifier(
            node, 
            n_input,
            n_output=output_len,
            classifier=type_classifier,
            sequential_kwargs=self.sequential_kwargs)
        


        if data_type == 'counts':
            x = relevant_cells[:, selected_var_names].X

        elif data_type == 'normlog':
            x = relevant_cells[:, selected_var_names].layers['normlog']

        elif data_type in relevant_cells.obsm:
            x = relevant_cells.obsm[data_type][:, selected_var_names]

        else:
            raise Exception('Data type not currently supported.')

        if hasattr(x, 'todense'):
            x = x.todense()

        y = np.array(relevant_cells.obs[children_obs_key])
        if hasattr(self, 'sampling_method') and type(self.sampling_method) != type(None):
            res = self.sampling_method(sampling_strategy=self.sampling_strategy)
            x, y = res.fit_resample(x, y)

        if "label_encoding" not in self.graph.nodes[node]:
            self.graph.nodes[node]['label_encoding'] = {}

        for label in child_nodes:
            if label in self.graph.nodes[node]['label_encoding']:
                continue

            idx = len(self.graph.nodes[node]['label_encoding'])
            self.graph.nodes[node]['label_encoding'][label] = idx

        y_int = np.array(
                [self.graph.nodes[node]['label_encoding'][label] for label in y]
            ).astype(int)
        y_onehot = keras.utils.to_categorical(y_int, num_classes=output_len)

        x = z_transform_properties(x)


        self.graph.nodes[node]['local_classifier']._train(x=x, y_onehot=y_onehot, y=y, y_int=y_int, train_kwargs=self.train_kwargs)


        timestamp = str(time()).replace('.', '_')
        if node not in self.trainings.keys():
            self.trainings[node] = {}

        self.trainings[node][timestamp] = {
            'barcodes': list(relevant_cells.obs_names),
            'node': node,
            'data_type': data_type,
            'type_classifier': type(self.graph.nodes[node].get('local_classifier', None)), #CAVE: get to not fail if ensemble is trained
            'var_names': selected_var_names
        }
        gc.collect()

        trained_node_params = {"local_classifier": self.graph.nodes[node]['local_classifier'],#.model,
                               "label_encoding": self.graph.nodes[node]['label_encoding'],
                               "selected_var_names": self.graph.nodes[node].get('selected_var_names', None)}

        #needed for used type of multiprocessing, return trained model to main process (is model universal for all used 
        # classifier types?)
        return trained_node_params

    def train_all_child_nodes_CPPN(
        self,
        current_node,
        train_barcodes=None,
        initial_call=True,
        parallelize = False, 
        ensemble_learning = False,
        **kwargs):
        """Starts at current_node, training its local classifier and following up by recursively
        training all local classifiers lower in the hierarchy. Uses cells that were labeled as 
        current_node at the relevant level (i. e. cells that are truly of type current_node, rather
        than cells predicted to be of that type). If train_barcodes is not None, only cells within
        that subset are used for training.

        Parameters
        ----------
        current_node
            When initially calling the method this refers to the node in the hierarchy at which
            to start training. Keeps track of which node is currently being trained over the course
            of the recursion.
        train_barcodes
            Barcodes (adata.obs_names) of those cells that are available for training of the
            classifier. Necessary to enable separation of training and test data for cross-validation.
        parallelize `bool`
            If `True` classifiers at all available nodes are trained in parallel. TODO: implement for ensemble learning
        ensemble_learning `bool`
            If `True`, an ensemble of classifiers is trained at each available node. 
        **kwargs 
            Keyword arguments to pass on to `train_single_node_CPPN` method. If ensemble_learning is `True`, the number 
            of classifiers per ensemble `n_ensemble_networks` (`int`) is mandatory.


        """

        if not parallelize:

            self.train_single_node_CPPN(current_node, train_barcodes, ensemble_learning=ensemble_learning, **kwargs)
            for child_node in self.get_child_nodes(current_node):
                if len(self.get_child_nodes(child_node)) == 0:
                    continue

                self.train_all_child_nodes_CPPN(child_node, 
                                                train_barcodes=train_barcodes, 
                                                ensemble_learning=ensemble_learning, 
                                                initial_call=False, 
                                                **kwargs)

            if initial_call:
                if "overall" not in self.trainings.keys():
                    self.trainings['overall'] = {}

                timestamp = str(time()).replace('.', '_')
                self.trainings['overall'][timestamp] = {
                    'train_barcodes': train_barcodes,
                    'current_node': current_node
                }

        else: 
            #initial_call not yet transferred

            print(f"Using multiprocessing for training with {mp.cpu_count()} available CPU cores.\n")
            if ensemble_learning:
                print(f"NOTE: ensemble learning currently not supported for parallel training." 
                      f"Defaulting to single classifier.\n")

            #find nodes that have to be trained be trained, i.e. nodes that have >= 2 child nodes 
            nodes_to_train = [node for node in list(self.graph.nodes()) if len(list(self.graph.successors(node))) >= 2] 
           
            with mp.Pool(processes=mp.cpu_count()) as pool: 
                all_trained_node_params = pool.map(self.train_single_node_CPPN, nodes_to_train)
            
            for node, params in zip(nodes_to_train, all_trained_node_params):
                if params is not None: #this should only happen at nodes that have not been trained
                    for key in params.keys():         
                        if params.get(key) is not None:    
                            self.graph.nodes[node][key] = params.get(key)


    def predict_single_node_CPPN_ensemble(
        self,
        node,
        barcodes=None,
        get_activations=False):
        """
        Predict cell labels using trained ensembles at the respective nodes. 

        TODO: merge with self.predict_single_node_CPPN possible/useful? Due to notation/new ensemble node attribute 
        not done for now  
        """

        print(f'Ensemble prediction at parent {node}.')

        #general sanity checks if node posseses trained models

        parent_obs_key = self.get_parent_obs_key(node)
        if f"{parent_obs_key}_pred" not in self.adata.obs.columns and barcodes is None and not node == self.root_node:
            raise Exception('If previous nodes were not predicted, barcodes for prediction need to \
                be given explicitly.')

        elif node == self.root_node and barcodes is None:
            barcodes = list(self.adata.obs_names)

        if not self.is_trained_at(node, ensemble = True):
            print(f'Must train local classifier or ensemble at {node} before trying to predict cell'\
                ' types')
            return

        if "label_encoding" not in self.graph.nodes[node] or len(self.graph.nodes[node]['label_encoding'].keys()) == 0:
            raise Exception('No label encoding saved in selected node. \
                The local classifier has either not been trained or the hierarchy updated and thus the output layer reset.')

        potential_cells = self.adata[barcodes, :]
        if f'{parent_obs_key}_pred' in self.adata.obs.columns:
            relevant_cells = potential_cells[potential_cells.obs[f'{parent_obs_key}_pred'] == node]

        else:
            relevant_cells = potential_cells

        if len(relevant_cells) == 0:
            return
        

        #prediction of single classifiers

        #check number of classifiers in ensemble 
        n_classifiers = len(self.graph.nodes[node]['trained_ensemble'].get('ensemble_classifier'))
        if n_classifiers == 0:
            print(f"No classifers trained in ensemble at node {node}.")
            return

        ensemble_predictions = []
        for network_i in range(n_classifiers):

            data_type = self.graph.nodes[node]['trained_ensemble']['ensemble_classifier'][network_i].data_type
            if type(self.graph.nodes[node]['trained_ensemble']['ensemble_classifier'][network_i]) not in [DenseKeras, DenseTorch, LogisticRegression]:
                raise Exception('CPPN classification mode currently only compatible with neural networks.')

            if data_type in ['counts', 'normlog']:
                selected_var_names = list(self.adata.var_names) 

            elif data_type in self.adata.obsm:
                selected_var_names = list(range(self.adata.obsm[data_type].shape[1]))

            else:
                raise Exception('Data type not supported.')

            # Feature selection is only relevant for (transformed) gene data, not embeddings, use saved features at node
            if self.use_feature_selection and 'selected_var_names' in self.graph.nodes[node]:
                selected_var_names = self.graph.nodes[node]['selected_var_names']

            if data_type == 'counts':
                x = relevant_cells[:, selected_var_names].X

            elif data_type == 'normlog':
                self.ensure_normlog()
                x = relevant_cells[:, selected_var_names].layers['normlog']

            elif data_type in relevant_cells.obsm:
                x = relevant_cells.obsm[data_type][:, selected_var_names]

            else:
                raise Exception('Data type not currently supported.')

            if hasattr(x, 'todense'):
                x = x.todense()

            x = z_transform_properties(x)

            #TODO: check updating activation return for ensemble
            # if get_activations:
            #     return list(relevant_cells.obs_names), self.graph.nodes[node]['local_classifier'].predict(x)

            y_pred_activations = self.graph.nodes[node]['trained_ensemble']['ensemble_classifier'][network_i].predict(x)
            ensemble_predictions.append(y_pred_activations)

            # if not self.prob_based_stopping:
            #     y_pred_activations = self.graph.nodes[node]['trained_ensemble']['ensemble_classifier'][network_i].predict(x)
            #     # y_pred_int = np.argmax(self.graph.nodes[node]['trained_ensemble']['ensemble_classifier'][network_i].predict(x), axis=-1)
                
            #     #save activations if network_i'th classifier
            #     ensemble_predictions.append(y_pred_activations)
        
        #%--------------------------------------------------------------------------------------------------------------------------------------------%#       
        #belongs somewhere in the prediction methds, not sure where yet because of test/training problem
        #%--------------------------------------------------------------------------------------------------------------------------------------------%#

        #calculate mean activation
        activations_ensemble_mean = np.mean(np.array([np.array(preds) for preds in ensemble_predictions]), axis = 0)
        #std not yet used - interesting to plot
        activations_ensemble_std = np.std(np.array([np.array(preds) for preds in ensemble_predictions]), axis = 0)

        # mean_activations = activations_sum / n_classifiers
        
        #=============================================================================#
        #NOTE CAVE: since all predictions are averaged, the label encoding of all trained classifiers has to be the same! 
        #=============================================================================#

        if not self.prob_based_stopping:

            y_pred_int = np.argmax(activations_ensemble_mean, axis=-1)

            #decode to class label
            label_decoding = {}
            for key in self.graph.nodes[node]['label_encoding'].keys():
                i = self.graph.nodes[node]['label_encoding'][key]
                label_decoding[i] = key

            y_pred = np.array(
                [label_decoding[i] for i in y_pred_int])
            #TODO: unproblematic method call?
            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, list(relevant_cells.obs_names), y_pred)

        elif self.prob_based_stopping:
            # y_pred = np.array(self.predict_single_node_proba(node, x))

            #NOTE: code below copied from self.predict_single_node_proba in hierarchy_base class

            #test if probability for one class is larger than threshold 
            largest_idx = np.argmax(activations_ensemble_mean, axis = -1) #TODO: np.asarray needed here? (see original method)
            if 'threshold' in self.graph.nodes[node]:
                threshold = self.graph.nodes[node]['threshold']

            else:
                threshold = self.threshold

            is_above_threshold = np.any(activations_ensemble_mean >= threshold, axis=1)
            largest_idx = largest_idx.astype(np.float32)
            largest_idx[~is_above_threshold] = np.nan

            #TODO after for loop?
            y_pred_nan_idx = np.where(np.isnan(y_pred))
            y_pred_not_nan_idx = np.where(~np.isnan(y_pred))
            y_pred = y_pred.astype(int)
            label_decoding = {}
            for key in self.graph.nodes[node]['label_encoding'].keys():
                i = self.graph.nodes[node]['label_encoding'][key]
                label_decoding[i] = key

            y_pred_not_nan_str = np.array(
                [label_decoding[i] for i in y_pred[y_pred_not_nan_idx]])
            y_pred = y_pred.astype(dtype="object") # important to avoid long cell types being truncated
            y_pred[y_pred_not_nan_idx] = y_pred_not_nan_str
            y_pred[y_pred_nan_idx] = 'stopped'
            #child_obs_key says at which hierarchy level the predictions have to be saved
            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, list(relevant_cells.obs_names), y_pred)


        if node not in self.predictions.keys():
            self.predictions[node] = {}

        timestamp = str(time()).replace('.', '_')
        self.predictions[node][timestamp] = {
            'barcodes': barcodes,
            'node': node,
            'data_type': self.graph.nodes[node]['local_classifier'].data_type,
            'type_classifier': type(self.graph.nodes[node]['local_classifier']),
            'var_names': selected_var_names,
        }
        gc.collect()

        if get_activations: 
            #for evaluation
            return ensemble_predictions, activations_ensemble_mean, activations_ensemble_std


        

    def predict_single_node_CPPN(
        self,
        node,
        barcodes=None,
        get_activations=False):
        """Uses an existing classifier at node to assign one of the child labels to the cells
        specified by barcodes. The predictions are stored in self.adata.obs by calling
        self.set_predictions under f'{obs_key}_pred' where obs_key is the key under
        which the annotations as one of the child labels are stored. If train_barcodes is not None, only cells within
        that subset are used for training.

        Parameters
        ----------
        node
            Specifies which local classifier is to be used for prediction. node='T' would result in
            predicting affiliation to one the T cell sub-labels.
        barcodes
            Specifies which cells are to be labelled.
        """

        print(f'Predicting at parent {node}.')

        parent_obs_key = self.get_parent_obs_key(node)
        if f"{parent_obs_key}_pred" not in self.adata.obs.columns and barcodes is None and not node == self.root_node:
            raise Exception('If previous nodes were not predicted, barcodes for prediction need to \
                be given explicitly.')

        elif node == self.root_node and barcodes is None:
            barcodes = list(self.adata.obs_names)

        if not self.is_trained_at(node):
            print(f'Must train local classifier for {node} before trying to predict cell'\
                ' types')
            return

        if "label_encoding" not in self.graph.nodes[node] or len(self.graph.nodes[node]['label_encoding'].keys()) == 0:
            raise Exception('No label encoding saved in selected node. \
                The local classifier has either not been trained or the hierarchy updated and thus the output layer reset.')

        potential_cells = self.adata[barcodes, :]
        if f'{parent_obs_key}_pred' in self.adata.obs.columns:
            relevant_cells = potential_cells[potential_cells.obs[f'{parent_obs_key}_pred'] == node]

        else:
            relevant_cells = potential_cells

        if len(relevant_cells) == 0:
            return

        data_type = self.graph.nodes[node]['local_classifier'].data_type
        if type(self.graph.nodes[node]['local_classifier']) not in [DenseKeras, DenseTorch, LogisticRegression]:
            raise Exception('CPPN classification mode currently only compatible with neural networks.')

        if data_type in ['counts', 'normlog']:
            selected_var_names = list(self.adata.var_names) 

        elif data_type in self.adata.obsm:
            selected_var_names = list(range(self.adata.obsm[data_type].shape[1]))

        else:
            raise Exception('Data type not supported.')

        # Feature selection is only relevant for (transformed) gene data, not embeddings
        if self.use_feature_selection and 'selected_var_names' in self.graph.nodes[node]:
            selected_var_names = self.graph.nodes[node]['selected_var_names']

        if data_type == 'counts':
            x = relevant_cells[:, selected_var_names].X

        elif data_type == 'normlog':
            self.ensure_normlog()
            x = relevant_cells[:, selected_var_names].layers['normlog']

        elif data_type in relevant_cells.obsm:
            x = relevant_cells.obsm[data_type][:, selected_var_names]

        else:
            raise Exception('Data type not currently supported.')

        if hasattr(x, 'todense'):
            x = x.todense()

        x = z_transform_properties(x)
        if get_activations:
            return list(relevant_cells.obs_names), self.graph.nodes[node]['local_classifier'].predict(x)

        if not self.prob_based_stopping:
            y_pred_int = np.argmax(self.graph.nodes[node]['local_classifier'].predict(x), axis=-1)
            label_decoding = {}
            for key in self.graph.nodes[node]['label_encoding'].keys():
                i = self.graph.nodes[node]['label_encoding'][key]
                label_decoding[i] = key

            y_pred = np.array(
                [label_decoding[i] for i in y_pred_int])
            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, list(relevant_cells.obs_names), y_pred)

        #%--------------------------------------------------------------------------------------------------------------------------------------------%#       
        #belongs somewhere in the prediction methds, not sure where yet because of test/training problem
        #%--------------------------------------------------------------------------------------------------------------------------------------------%#

        elif self.prob_based_stopping:
            y_pred = np.array(self.predict_single_node_proba(node, x))
            y_pred_nan_idx = np.where(np.isnan(y_pred))
            y_pred_not_nan_idx = np.where(~np.isnan(y_pred))
            y_pred = y_pred.astype(int)
            label_decoding = {}
            for key in self.graph.nodes[node]['label_encoding'].keys():
                i = self.graph.nodes[node]['label_encoding'][key]
                label_decoding[i] = key

            y_pred_not_nan_str = np.array(
                [label_decoding[i] for i in y_pred[y_pred_not_nan_idx]])
            y_pred = y_pred.astype(dtype="object") # important to avoid long cell types being truncated
            y_pred[y_pred_not_nan_idx] = y_pred_not_nan_str
            y_pred[y_pred_nan_idx] = 'stopped'
            #child_obs_key says at which hierarchy level the predictions have to be saved
            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, list(relevant_cells.obs_names), y_pred)

        if node not in self.predictions.keys():
            self.predictions[node] = {}

        timestamp = str(time()).replace('.', '_')
        self.predictions[node][timestamp] = {
            'barcodes': barcodes,
            'node': node,
            'data_type': self.graph.nodes[node]['local_classifier'].data_type,
            'type_classifier': type(self.graph.nodes[node]['local_classifier']),
            'var_names': selected_var_names,
        }
        gc.collect()

    def predict_all_child_nodes_CPPN(
        self,
        current_node,
        current_barcodes=None,
        initial_call=True,
        use_ensemble_prediction=False,
        **ensemble_kwargs):

        if type(current_barcodes) == type(None):
            current_barcodes = self.adata.obs_names

        if len(current_barcodes) == 0:
            return

        if not self.is_trained_at(current_node):
            return
        
        if use_ensemble_prediction:
            self.predict_single_node_CPPN_ensemble(current_node, barcodes=current_barcodes, **ensemble_kwargs)
        
        else:
            self.predict_single_node_CPPN(current_node, barcodes=current_barcodes)
        obs_key = self.get_children_obs_key(current_node)
        for child_node in self.get_child_nodes(current_node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            try:
                child_node_barcodes = self.get_predicted_barcodes(
                    obs_key, 
                    child_node)
                self.predict_all_child_nodes_CPPN(child_node, current_barcodes=child_node_barcodes, initial_call=False)
            except KeyError:
                print(f'Tried to predict children of {current_node}, current_barcodes is {current_barcodes}')   
                break         

        if initial_call:
            if "overall" not in self.predictions.keys():
                self.predictions['overall'] = {}

            timestamp = str(time()).replace('.', '_')
            self.predictions['overall'][timestamp] = {
                'current_barcodes': current_barcodes,
                'current_node': current_node
            }

    def update_hierarchy_CPPN(self, dict_of_cell_relations, root_node=None):
        if root_node is not None:
            self.root_node = root_node

        if dict_of_cell_relations == self.dict_of_cell_relations:
            return

        self.ensure_depth_match(dict_of_cell_relations, self.obs_names)
        self.ensure_unique_nodes(dict_of_cell_relations)
        all_nodes_pre = flatten_dict(self.dict_of_cell_relations)
        self.dict_of_cell_relations = dict_of_cell_relations
        all_nodes_post = flatten_dict(self.dict_of_cell_relations)
        self.all_nodes = all_nodes_post
        self.node_to_depth = set_node_to_depth(self.dict_of_cell_relations)
        new_graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, new_graph)

        new_nodes = [n for n in all_nodes_post if n not in all_nodes_pre]
        [n for n in all_nodes_pre if n not in all_nodes_post]
        moved_nodes = []
        classifier_nodes = []
        for node in all_nodes_post:
            if node in new_nodes:
                continue

            # Check if node was moved within the hierarchy, i. e. assigned
            # to a different parent node
            # Does not change the strategy of assigning the previous node attributes
            # but may end up a fact of interest
            if not node == self.root_node:
                parent_post = self.get_parent_node(node, graph=new_graph)
                parent_pre = self.get_parent_node(node)
                if parent_pre != parent_post:
                    moved_nodes.append(node)

            # Transfer properties, such as local classifier, from old graph
            # to new graph
            for item in self.graph.nodes[node]:
                new_graph.nodes[node][item] = deepcopy(self.graph.nodes[node][item])

            if "local_classifier" in self.graph.nodes[node]:
                classifier_nodes.append(node) # Define nodes that contain a classifier

            print(f'Transfered to {node}, local classifier {"transferred" if "local_classifier" in self.graph.nodes[node] else "not transferred"}')

        self.graph = new_graph
        for node in classifier_nodes:
            print(f'Ensuring correct output architecture for {node}.')
            child_nodes = self.get_child_nodes(node)
            # Previously reset all classifier nodes
            # Bad idea because you want to conserve as much of the training progress as possible,
            # resetting as little as possible, as much as necessary
            if True in [n in new_nodes or n in moved_nodes for n in [node] + list(child_nodes)]:
                if type(self.graph.nodes[node]['local_classifier']) is LogisticRegression:
                    print('Cannot adjust output structure of logistic regression classifier.')
                    continue

                # reset label encoding, unproblematic because the final layer is reinitilaized anyway
                self.graph.nodes[node]['label_encoding'] = {}
                self.graph.nodes[node]['local_classifier'].reset_output(len(child_nodes))