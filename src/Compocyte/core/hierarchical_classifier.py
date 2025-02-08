from typing import Union
from sklearn.feature_selection import SelectKBest, f_classif
from Compocyte.core.base.data_base import DataBase
from Compocyte.core.base.hierarchy_base import HierarchyBase
from Compocyte.core.base.export_import_base import ExportImportBase
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.fit_methods import fit, predict
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.models.dense_torch import DenseTorch
from time import time
import numpy as np
import os
import pickle
import scanpy as sc
import multiprocessing as mp

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
                    if isinstance(local_classifier, DenseTorch) or isinstance(local_classifier, LogisticRegression) or isinstance(local_classifier, DummyClassifier):
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

                elif 'labels_dec.pickle' in contents:
                    classifier = LogisticRegression._load(os.path.join(model_path, last_timestamp))

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

    def select_subset(self, node: str, features: list=None) -> sc.AnnData:
        obs = self.obs_names[self.node_to_depth[node]]
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        is_node = self.adata.obs[obs] == node
        has_child_label = self.adata.obs[child_obs] != ''
        subset = self.adata[is_node & has_child_label]
        if features is not None:
            subset = subset[:, features]

        return subset
    
    def select_subset_prediction(self, node: str, features: list=None) -> sc.AnnData:
        obs = self.obs_names[self.node_to_depth[node]]
        obs = f'{obs}_pred'
        if obs not in self.adata.obs.columns:
            subset = self.adata

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
            test_factor: float=1.0):
        
        has_features = 'selected_var_names' in self.graph.nodes[node].keys()
        if has_features and not overwrite:
            raise Exception(f'Features have already been selected at {node}.')
        
        subset = self.select_subset(node)
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

        x = np.array(subset.X)
        x = z_transform_properties(x)   
        y = np.array(subset.obs[child_obs])
        selecter = SelectKBest(f_classif, k=n_features)
        selecter.fit(x, y)
        features = self.adata.var_names[selecter.get_support()]

        return features.tolist()

    def create_local_classifier(
            self, 
            node: str,
            overwrite: bool=False,
            classifier_type: Union[DenseTorch, LogisticRegression]=DenseTorch,
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
        if n_output == 1:
            classifier_type = DummyClassifier

        local_classifier = classifier_type(
            labels,
            n_input=n_input,
            n_output=n_output,
            **classifier_kwargs)
        self.graph.nodes[node]['local_classifier'] = local_classifier


    def train_single_node(self, node, **fit_kwargs):
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
            }
            self.graph.nodes[node]['threshold'] = kwargs['threshold']

        else:
            features_kwargs = {}
            classifier_kwargs = {}

        if not has_classifier:
            subset = self.select_subset(node)
            if len(subset) == 0:
                return
            
            features = self.run_feature_selection(node, **features_kwargs)
            self.graph.nodes[node]['selected_var_names'] = features
            self.create_local_classifier(node, **classifier_kwargs)
        
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        features = self.graph.nodes[node]['selected_var_names']
        subset = self.select_subset(node, features=features)
        if len(subset) == 0:
                return
        
        model = self.graph.nodes[node]['local_classifier']
        x = subset.X
        y = subset.obs[child_obs].values
        print(f'Training at {node}.')

        # Necessary to avoid data loss when using mp.pool
        return {
            **self.graph.nodes[node],
            'learning_curve': fit(model, x, y, **fit_kwargs)
        }

    def train_all_child_nodes(
        self,
        parallelize: bool=False) -> None:

        nodes_to_train = []
        for node in self.graph.nodes:
            n_children = len(list(self.graph.successors(node)))
            if n_children >= 1:
                nodes_to_train.append(node)

        if not parallelize:
            for node in nodes_to_train:
                self.train_single_node(node, parallelize=False)

        else: 
            #initial_call not yet transferred
            # When setting num_threads > 1 per training process, the number of processes should be limited
            if self.num_threads is None:
                processes = mp.cpu_count()

            else:
                processes = int(mp.cpu_count() / self.num_threads)

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
        threshold: float=-1) -> np.array:

        if 'local_classifier' not in self.graph.nodes[node]:
            return []
        
        features = self.graph.nodes[node]['selected_var_names']
        subset = self.select_subset_prediction(node, features=features)
        if len(subset) == 0:
            return
        
        model = self.graph.nodes[node]['local_classifier']
        x = subset.X
        print(f'Predicting at {node}.')

        pred = predict(model, x, threshold=threshold)
        child_obs = self.obs_names[self.node_to_depth[node] + 1] 
        child_obs = f'{child_obs}_pred'
        if child_obs not in self.adata.obs.columns:
            self.adata.obs[child_obs] = ''
            
        self.adata.obs[child_obs] = self.adata.obs[child_obs].astype(str)
        self.adata.obs.loc[
            subset.obs_names,
            child_obs
        ] = pred

        return pred    

    def predict_all_child_nodes(
        self, 
        node: str,
        threshold: float=-1,
        mlnp: bool=False):

        # For mandatory leaf node prediction use -1
        if not mlnp:
            threshold = self.graph.nodes[node].get('threshold', threshold)

        self.predict_single_node(node, threshold=threshold)
        for child_node in self.get_child_nodes(node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            self.predict_all_child_nodes(child_node, threshold=threshold, mlnp=mlnp)

