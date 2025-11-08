from typing import Union
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import robust_scale
from Compocyte.core.base.data_base import DataBase
from Compocyte.core.base.hierarchy_base import HierarchyBase
from Compocyte.core.base.export_import_base import ExportImportBase
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.fit_methods import fit, predict
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.models.dense_torch import DenseTorch
from time import time
from scipy import sparse
import numpy as np
import os
import pickle
import scanpy as sc
import multiprocessing as mp
from Compocyte.core.models.trees import BoostedTrees
from Compocyte.core.tools import z_transform_properties


class HierarchicalClassifier(
        DataBase,
        HierarchyBase,
        ExportImportBase):

    def __init__(
            self,
            save_path,
            adata=None,
            root_node=None,
            dict_of_cell_relations=None, 
            obs_names=None,
            default_input_data='normlog',
            num_threads=None,
            ignore_counts=False,
            ):

        self.save_path = save_path
        self.default_input_data = default_input_data
        self.num_threads = num_threads
        self.adata = None
        self.var_names = None
        self.dict_of_cell_relations = None
        self.root_node = None
        self.obs_names = None
        self.ignore_counts = ignore_counts

        if type(adata) != type(None):
            self.load_adata(adata)

        if root_node is not None and dict_of_cell_relations is not None and obs_names is not None:
            self.set_cell_relations(root_node, dict_of_cell_relations, obs_names)

    def save(self, save_adata=False):
        # save all attributes
        # get types, for adata use adatas write function with hash of adata
        # save state of all local classifiers (what does dumping self.graph do?)
        # save state of all nodes in the graph, label encoders, var names ...
        data_path = os.path.join(
            self.save_path, 
            'data'
        )
        timestamp = str(time()).replace('.', '_')
        hc_path = os.path.join(
            self.save_path, 
            'hierarchical_classifiers',
            timestamp
        )
        if not os.path.exists(hc_path):
            os.makedirs(hc_path)

        settings_dict = {}
        for key in self.__dict__.keys():
            if key == 'adata':
                if self.adata is None or not save_adata:
                    continue
                
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                self.adata.write(os.path.join(data_path, f'{timestamp}.h5ad'))

            elif key == 'graph':
                continue

            else:
                settings_dict[key] = self.__dict__[key]

        with open(os.path.join(hc_path, 'hierarchical_classifier_settings.pickle'), 'wb') as f:
            pickle.dump(settings_dict, f)
        
        for node in list(self.graph):
            node_content_path = os.path.join(
                self.save_path, 
                'node_content',
                node,
                timestamp
            )
            if not os.path.exists(node_content_path):
                os.makedirs(node_content_path)

            for key in self.graph.nodes[node].keys():
                if key == 'local_classifier':
                    model_path = os.path.join(
                        self.save_path, 
                        'models',
                        node,
                        timestamp
                    )
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)

                    local_classifier = self.graph.nodes[node]['local_classifier']
                    if isinstance(local_classifier, DenseTorch) or isinstance(local_classifier, LogisticRegression) or isinstance(local_classifier, DummyClassifier) or isinstance(local_classifier, BoostedTrees):
                        self.graph.nodes[node]['local_classifier']._save(model_path)
                        
                    continue

                with open(os.path.join(node_content_path, f'{key}.pickle'), 'wb') as f:
                    pickle.dump(self.graph.nodes[node][key], f)

    def load(self, load_path=None, load_adata=False):
        if load_path is None:
            load_path = self.save_path
            
        data_path = os.path.join(
            load_path, 
            'data'
        )
        hc_path = os.path.join(
            load_path, 
            'hierarchical_classifiers'
        )
        if os.path.exists(hc_path):
            timestamps = os.listdir(hc_path)
            last_timestamp = sorted(timestamps)[-1]
            with open(os.path.join(hc_path, last_timestamp, 'hierarchical_classifier_settings.pickle'), 'rb') as f:
                settings_dict = pickle.load(f)
                for key in settings_dict.keys():
                    self.__dict__[key] = settings_dict[key]

        if os.path.exists(data_path) and load_adata:
            timestamps = os.listdir(data_path)
            last_adata = sorted(timestamps)[-1]
            adata = sc.read_h5ad(os.path.join(data_path, last_adata))
            self.load_adata(adata)

        if not hasattr(self, 'graph') or self.graph is None:
            self.make_classifier_graph()

        for node in list(self.graph):
            model_path = os.path.join(
                load_path, 
                'models',
                node
            )
            node_content_path = os.path.join(
                load_path, 
                'node_content',
                node
            )
            if os.path.exists(model_path):
                timestamps = os.listdir(model_path)
                last_timestamp = sorted(timestamps)[-1]
                contents = os.listdir(os.path.join(model_path, last_timestamp))
                if len([c for c in contents if c.startswith('non_param_dict')]) > 0:
                    classifier = DenseTorch._load(os.path.join(model_path, last_timestamp))

                elif 'labels_dec.pickle' in contents and not 'model.cbm' in contents:
                    classifier = LogisticRegression._load(os.path.join(model_path, last_timestamp))

                elif 'labels_dec.pickle' in contents and 'model.cbm' in contents:
                    classifier = BoostedTrees._load(os.path.join(model_path, last_timestamp))

                else:
                    classifier = DummyClassifier._load(os.path.join(model_path, last_timestamp))

                self.graph.nodes[node]['local_classifier'] = classifier

            if os.path.exists(node_content_path):
                timestamps = os.listdir(node_content_path)
                last_timestamp = sorted(timestamps)[-1]
                properties = os.listdir(os.path.join(node_content_path, last_timestamp))
                for p in properties:
                    key = p.replace('.pickle', '')
                    with open(
                        os.path.join(node_content_path, last_timestamp, p), 
                        'rb'
                    ) as f:
                        p = pickle.load(f)
                        self.graph.nodes[node][key] = p

    def limit_cells(
            self,
            subset: sc.AnnData, 
            max_cells: int, 
            stratify_by: str) -> sc.AnnData:
        
        if len(subset) > max_cells:
            rng = np.random.default_rng(42)
            datasets = subset.obs[stratify_by].unique()
            cells_per_dataset = max_cells // len(datasets)
            limited_indices = []

            for dataset in datasets:
                dataset_indices = subset.obs[subset.obs['dataset'] == dataset].index
                if len(dataset_indices) > cells_per_dataset:
                    dataset_indices = rng.choice(dataset_indices, cells_per_dataset, replace=False)
                    
                limited_indices.extend(dataset_indices)

            # If we have fewer cells than max_cells, randomly sample additional cells to reach max_cells
            if len(limited_indices) < max_cells:
                additional_indices = rng.choice(
                    subset.obs.index.difference(limited_indices), 
                    max_cells - len(limited_indices), 
                    replace=False)
                limited_indices.extend(additional_indices)

            subset = subset[limited_indices, :]

        return subset

    def introduce_limit(self, max_cells: int, stratify_by: str):
        """
        Introduces a limit on the number of cells per local classifier training and \
            specifies a stratification criterion.

        Parameters:
        max_cells (int): The maximum number of cells allowed.
        stratify_by (str): The criterion by which to stratify the cells.
        """

        self.max_cells = max_cells
        self.stratify_by = stratify_by

    def select_subset(
            self, 
            node: str, 
            features: list=None,
            max_cells: int=None) -> sc.AnnData:
        
        obs = self.obs_names[self.node_to_depth[node]]
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        is_node = self.adata.obs[obs] == node
        has_child_label = self.adata.obs[child_obs] != ''
        subset = self.adata[is_node & has_child_label]
        if features is not None:
            subset = subset[:, features]

        stratify_by = getattr(self, 'stratify_by', None)
        if max_cells is not None and stratify_by is not None:
            subset = self.limit_cells(subset, max_cells, stratify_by)

        """
        if max_cells is None:
            max_cells = getattr(self, 'max_cells', None)

        stratify_by = getattr(self, 'stratify_by', None)
        if max_cells is not None and stratify_by is not None:
            subset = self.limit_cells(subset, max_cells, stratify_by)

        """

        return subset
    
    def select_subset_prediction(self, node: str, features: list=None, for_trial=False) -> sc.AnnData:
        obs = self.obs_names[self.node_to_depth[node]]
        obs = f'{obs}_pred'
        if obs not in self.adata.obs.columns and not for_trial:
            subset = self.adata

        elif obs not in self.adata.obs.columns and for_trial:
            is_node = self.adata.obs[self.obs_names[self.node_to_depth[node]]] == node
            subset = self.adata[is_node]

        else:
            is_node = self.adata.obs[obs] == node
            subset = self.adata[is_node]

        if features is not None:
            subset = subset[:, features]

        return subset

    def run_feature_selection(
            self,
            node: str,
            overwrite: bool=False,
            n_features: int=-1,
            max_features: int=None,
            min_features: int=30,
            test_factor: float=1.0,
            max_cells=100_000):
        
        has_features = 'selected_var_names' in self.graph.nodes[node].keys()
        if has_features and not overwrite:
            raise Exception(f'Features have already been selected at {node}.')
        
        subset = self.select_subset(node, max_cells=max_cells)
        x = sparse.csr_matrix.toarray(subset.X)
            
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        if len(subset.obs[child_obs].unique()) <= 1:
            return self.adata.var_names.tolist()
        
        # Rule of thumb from Google's rules of ML:
        # At least 100 samples per feature
        if n_features < 0:
            # test_factor should be taken account during hypopt with reduced sample numbers
            n_features = int(len(subset) / 100 * test_factor)

        n_features = max(min_features, n_features)
        if max_features is None:
            max_features = len(self.adata.var_names)

        n_features = min(n_features, max_features)

        x = np.asarray(x)
        x = robust_scale(x, axis=1, with_centering=False, copy=False, unit_variance=True)
        y = np.array(subset.obs[child_obs])
        selecter = SelectKBest(f_classif, k=n_features)
        selecter.fit(x, y)
        features = self.adata.var_names[selecter.get_support()]

        return features.tolist()

    def create_local_classifier(
            self, 
            node: str,
            overwrite: bool=False,
            classifier_type: Union[DenseTorch, LogisticRegression, BoostedTrees]=DenseTorch,
            **classifier_kwargs):
        
        has_classifier = 'local_classifier' in self.graph.nodes[node].keys()
        if has_classifier and not overwrite:
            raise Exception(f'A classifier already exists at {node}.')
        
        features = self.graph.nodes[node].get('selected_var_names', None)
        if features is None:
            raise Exception(f'Cannot create classifier at {node} without features.\
                            Please run run_feature_selection first.')
        
        subset = self.select_subset(node)
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        labels = subset.obs[child_obs].unique().tolist()
        n_input = len(features)
        n_output = len(labels)
        if isinstance(classifier_type, str):
            if classifier_type == 'DenseTorch':
                classifier_type = DenseTorch

            elif classifier_type == 'LogisticRegression':
                classifier_type = LogisticRegression

            elif classifier_type == 'BoostedTrees':
                classifier_type = BoostedTrees

            else:
                raise Exception(f'Unknown classifier type: {classifier_type}')
            
        if n_output == 1:
            classifier_type = DummyClassifier

        local_classifier = classifier_type(
            labels,
            n_input=n_input,
            n_output=n_output,
            **classifier_kwargs)
        self.graph.nodes[node]['local_classifier'] = local_classifier


    def train_single_node(self, node, standardize_separately: str=None, **fit_kwargs):
        if not hasattr(self, 'num_threads') and not 'num_threads' in fit_kwargs:
            raise Exception('Please specify the number of threads to use for training.')
        
        elif 'num_threads' in fit_kwargs:
            self.num_threads = fit_kwargs['num_threads']

        has_classifier = 'local_classifier' in self.graph.nodes[node].keys()
        # This weird approach is currently necessary to allow for training with mp.pool
        if hasattr(self, 'tuned_kwargs') and node in self.tuned_kwargs:
            kwargs = self.tuned_kwargs[node]
            features_kwargs = {
                'n_features': kwargs['n_features']
            }
            classifier_kwargs = {
                'hidden_layers': kwargs['hidden_layers'],
                'dropout': kwargs['dropout'],
            }
            fit_kwargs = {
                'epochs': kwargs['epochs'],
                'batch_size': kwargs['batch_size'],
                'starting_lr': kwargs['starting_lr'],
                'max_lr': kwargs['max_lr'],
                'momentum': kwargs['momentum'],
                'beta': kwargs['beta'],
                'gamma': kwargs['gamma'],
                'max_cells': getattr(self, 'max_cells', 1_000_000)
            }
            self.graph.nodes[node]['threshold'] = kwargs['threshold']

        else:
            features_kwargs = {}
            classifier_kwargs = {}

        if not has_classifier:
            subset = self.select_subset(node)
            if len(subset) < 5:
                return
            
            features = self.run_feature_selection(node, **features_kwargs)
            self.graph.nodes[node]['selected_var_names'] = features
            classifier_type = DenseTorch
            hidden_layers = classifier_kwargs.get('hidden_layers', [])
            if -1 in hidden_layers:
                classifier_type = BoostedTrees
                
            # If classifier types other than the standard have been set, use those
            specified_classifier_types =  getattr(self, 'specified_classifier_types', {})
            classifier_type = specified_classifier_types.get(node, classifier_type)
            self.create_local_classifier(node, classifier_type=classifier_type, **classifier_kwargs)
        
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        features = self.graph.nodes[node]['selected_var_names']
        subset = self.select_subset(node, features=features)
        if len(subset) == 0:
                return
        
        model = self.graph.nodes[node]['local_classifier']
        x = subset.X
        y = subset.obs[child_obs].values
        print(f'Training at {node}.')
        
        if standardize_separately is not None:
            idx = []
            for dataset in subset.obs[standardize_separately].unique():
                idx.append(np.where(subset.obs[standardize_separately] == dataset))

        else:
            idx = None
            
        if not 'max_cells' in fit_kwargs:
            fit_kwargs['max_cells'] = getattr(self, 'max_cells', 1_000_000)

        if not 'num_threads' in fit_kwargs:
            fit_kwargs['num_threads'] = self.num_threads
            
        # Necessary to avoid data loss when using mp.pool
        return {
            **self.graph.nodes[node],
            'learning_curve': fit(model, x, y, standardize_idx=idx, **fit_kwargs)
        }
    
    def set_classifier_type(self, node, classifier_type):
        if isinstance(node, list):
            for n in node:
                self.set_classifier_type(n, classifier_type)

        else:
            if not hasattr(self, 'specified_classifier_types'):
                self.specified_classifier_types = {}

            self.specified_classifier_types[node] = classifier_type

    def train_all_child_nodes(
        self,
        parallelize: bool=False,
        processes: int=None) -> None:
        
        nodes_to_train = []
        for node in self.graph.nodes:
            n_children = len(list(self.graph.successors(node)))
            if n_children >= 1:
                nodes_to_train.append(node)

        if not parallelize:
            for node in nodes_to_train:
                self.train_single_node(node, parallelize=False)

        else: 
            if processes is None:
                raise Exception('Please specify the number of processes to use for parallelization.')
            
            # When setting num_threads > 1 per training process, the number of processes should be limited
            if self.num_threads is not None:
                processes = int(processes / self.num_threads)

            print(f"Using multiprocessing for training with {mp.cpu_count()} available CPU cores.\n")           
            with mp.Pool(processes=processes) as pool: 
                all_trained_node_params = pool.map(self.train_single_node, nodes_to_train)
            
            for node, params in zip(nodes_to_train, all_trained_node_params):
                if params is not None: #this should only happen at nodes that have not been trained
                    for key in params.keys():         
                        if params.get(key) is not None:    
                            self.graph.nodes[node][key] = params.get(key)

    def predict_single_node(
        self, 
        node: str,
        threshold: float=-1,
        monte_carlo: int=None) -> np.array:

        if 'local_classifier' not in self.graph.nodes[node]:
            return []
        
        features = self.graph.nodes[node]['selected_var_names']
        subset = self.select_subset_prediction(node, features=features)
        if len(subset) == 0:
            return
        
        model = self.graph.nodes[node]['local_classifier']
        x = subset.X
        print(f'Predicting at {node}.')

        pred = predict(model, x, threshold=threshold, monte_carlo=monte_carlo)
        all_logits = None
        if monte_carlo is not None and isinstance(pred, tuple):
            pred, all_logits = pred
            if len(all_logits.shape) < 3:
                all_logits = np.expand_dims(all_logits, axis=1)

        
        if 'overclustering' in subset.obs.columns:
            for cluster_name in subset.obs['overclustering'].unique():
                cluster_indices = subset.obs['overclustering'] == cluster_name
                if np.sum(cluster_indices) > 0:
                    cluster_preds = pred[cluster_indices]
                    if len(cluster_preds) > 0:
                        most_common = max(set(cluster_preds), key=list(cluster_preds).count)
                        pred[cluster_indices] = most_common

        child_obs = self.obs_names[self.node_to_depth[node] + 1] 
        child_obs = f'{child_obs}_pred'
        if child_obs not in self.adata.obs.columns:
            self.adata.obs[child_obs] = ''
            
        self.adata.obs[child_obs] = self.adata.obs[child_obs].astype(str)
        self.adata.obs.loc[
            subset.obs_names,
            child_obs
        ] = pred
        if monte_carlo is not None and all_logits is not None:
            # all_logits: Shape: (iterations, samples, labels)
            # mean_activations_per_sample: Shape: (samples, labels)
            mean_activations_per_sample = np.mean(all_logits, axis=0)
            # idx_max_activations: Shape: (samples)
            idx_max_activations = np.argmax(mean_activations_per_sample, axis=1)
            # idx_tile: Shape: (iterations, samples)
            idx_tile = np.tile(idx_max_activations, (all_logits.shape[0], 1))
            # Expand idx_tile dims to match test dimensions
            # For each iteration and sample, take the activation corresponding to
            # the label with the highest mean activation across iterations for this sample
            # activations_chosen_label_per_iteration: Shape: (iterations, samples, 1)
            activations_chosen_label_per_iteration = np.take_along_axis(
                all_logits, 
                np.expand_dims(idx_tile, axis=2), axis=2)
            # activations_chosen_label_per_sample: Shape: (samples, iterations)
            activations_chosen_label_per_sample = np.squeeze(activations_chosen_label_per_iteration).T
            # when dealing with single samples, activations are squeezed to 1D
            axis = 0 if activations_chosen_label_per_sample.ndim == 1 else 1
            # mean_activation_chosen_label_per_sample: Shape: (samples)
            mean_activation_chosen_label_per_sample = np.mean(activations_chosen_label_per_sample, axis=axis)
            std_activations_chosen_label_per_sample = np.std(activations_chosen_label_per_sample, axis=axis)
            self.adata.obs.loc[
                subset.obs_names,
                'monte_carlo_mean',
            ] = mean_activation_chosen_label_per_sample
            self.adata.obs.loc[
                subset.obs_names,
                'monte_carlo_std',
            ] = std_activations_chosen_label_per_sample

        return pred    

    def predict_all_child_nodes(
        self, 
        node: str,
        threshold: float=-1,
        mlnp: bool=False,
        monte_carlo: int=None):

        # For mandatory leaf node prediction use -1
        if not mlnp:
            threshold = self.graph.nodes[node].get('threshold', threshold)

        self.predict_single_node(node, threshold=threshold, monte_carlo=monte_carlo)
        for child_node in self.get_child_nodes(node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            self.predict_all_child_nodes(child_node, threshold=threshold, mlnp=mlnp, monte_carlo=monte_carlo)

