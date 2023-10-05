from scFlorist.core.tools import z_transform_properties
from scFlorist.core.models.dense import DenseKeras
from scFlorist.core.models.log_reg import LogisticRegression
from scFlorist.core.models.dense_torch import DenseTorch
from scFlorist.core.tools import flatten_dict, make_graph_from_edges, set_node_to_depth
from time import time
from copy import deepcopy
import tensorflow.keras as keras
import numpy as np
import networkx as nx

class CPNBase():
    def train_single_node_CPN(
        self,
        node,
        train_barcodes=None):

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
        
        parent_node = self.get_parent_node(node)
        parent_obs_key = self.get_parent_obs_key(parent_node)
        child_nodes = self.get_child_nodes(node)
        node_obs_key = self.get_children_obs_key(parent_node)
        # We use the sibling policy for defining positive and negative training samples
        # for a 1 vs all classifier. All cells that are classified at the appropriate
        # level as parent_node are siblings of node.
        is_parent_node = self.adata.obs[parent_obs_key] == parent_node
        is_labeled = self.adata.obs[node_obs_key].isin(child_nodes)
        potential_cells = self.adata[is_parent_node & is_labeled]
        # To avoid training with cells that should be reserved for testing, make sure to limit
        # to train_barcodes.
        relevant_cells = self.adata[[b for b in potential_cells.obs_names if b in train_barcodes], :]
        # Cells that are unlabeled (nan or '') are not certain to not belong to the cell type in question
        positive_cells = relevant_cells[relevant_cells.obs[node_obs_key] == node]
        negative_cells = relevant_cells[relevant_cells.obs[node_obs_key] != node]
        if data_type == 'normlog':
            self.ensure_normlog()

        if len(positive_cells) == 0:
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

        elif self.use_feature_selection:
            pct_relevant = len(relevant_cells) / len(self.adata)
            # n_features is simply overwritten if method=='hvg'
            projected_relevant_cells = pct_relevant * self.projected_total_cells
            # should not exceed a ratio of 1:100 of features to training samples as per google rules of ml
            n_features_by_samples = int(projected_relevant_cells / 100)
            n_features = max(
                min(self.max_features, n_features_by_samples),
                self.min_features
            )
            # Counts and normlog rely on variable names for unique id of selected features
            # for embeddings, only selected indices can be provided
            return_idx = data_type not in ['counts', 'normlog']
            selected_var_names = self.feature_selection(
                list(positive_cells.obs_names), 
                list(negative_cells.obs_names), 
                data_type, 
                n_features=n_features, 
                method='f_classif',
                return_idx=return_idx)

        # Make sure that selected var names are exported even if all features are used
        # for export compatibility
        self.graph.nodes[node]['selected_var_names'] = selected_var_names
        n_input = len(selected_var_names)
        self.ensure_existence_OVR_classifier(
            node, 
            n_input,
            type_classifier,
            data_type,
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

        y = np.array(relevant_cells.obs[node_obs_key])
        if hasattr(self, 'sampling_method') and type(self.sampling_method) != type(None):
            res = self.sampling_method(sampling_strategy=self.sampling_strategy)
            x, y = res.fit_resample(x, y)

        y_int = (y == node).astype(int)
        y_onehot = keras.utils.to_categorical(y_int, num_classes=2)
        x = z_transform_properties(x)
        self.graph.nodes[node]['local_classifier']._train(x=x, y_onehot=y_onehot, y=y, y_int=y_int, train_kwargs=self.train_kwargs)
        timestamp = str(time()).replace('.', '_')
        if node not in self.trainings.keys():
            self.trainings[node] = {}

        self.trainings[node][timestamp] = {
            'barcodes': list(relevant_cells.obs_names),
            'node': node,
            'data_type': self.graph.nodes[node]['local_classifier'].data_type,
            'type_classifier': type(self.graph.nodes[node]['local_classifier']),
            'var_names': selected_var_names,
        }

    def train_all_child_nodes_CPN(
        self,
        node,
        train_barcodes=None,
        initial_call=True):
        """"""

        if train_barcodes is None:
            train_barcodes = list(self.adata.obs_names)

        # Can not train root node classifier as there are no negative cells
        if not node == self.root_node:
            self.train_single_node_CPN(node, train_barcodes)

        for child_node in self.get_child_nodes(node):
            self.train_all_child_nodes_CPN(child_node, train_barcodes=train_barcodes, initial_call=False)

        if initial_call:
            if "overall" not in self.trainings.keys():
                self.trainings['overall'] = {}

            timestamp = str(time()).replace('.', '_')
            self.trainings['overall'][timestamp] = {
                'train_barcodes': train_barcodes,
                'current_node': node
            }

    def predict_single_parent_node_CPN(self, node, barcodes=None, get_activations=False):
        """"""

        print(f'Predicting at parent {node}.')

        parent_obs_key = self.get_parent_obs_key(node)
        if f"{parent_obs_key}_pred" not in self.adata.obs.columns and barcodes is None and not node == self.root_node:
            raise Exception('If previous nodes were not predicted, barcodes for prediction need to \
                be given explicitly.')

        elif node == self.root_node and barcodes is None:
            barcodes = list(self.adata.obs_names)

        potential_cells = self.adata[barcodes, :]

        if f'{parent_obs_key}_pred' in self.adata.obs.columns:
            relevant_cells = potential_cells[potential_cells.obs[f'{parent_obs_key}_pred'] == node]

        else:
            relevant_cells = potential_cells

        if len(relevant_cells) == 0:
            return

        activations_positive = None
        predicted_nodes = []
        child_nodes = self.get_child_nodes(node)
        for child_node in child_nodes:
            if "local_classifier" not in self.graph.nodes[child_node].keys():
                continue

            predicted_nodes.append(child_node)
            data_type = self.graph.nodes[child_node]['local_classifier'].data_type
            if type(self.graph.nodes[child_node]['local_classifier']) not in [DenseKeras, DenseTorch, LogisticRegression]:
                raise Exception('CPN classification mode currently only compatible with neural networks.')

            selected_var_names = list(self.adata.var_names)
            # Feature selection is only relevant for (transformed) gene data, not embeddings
            if self.use_feature_selection and data_type in ['counts', 'normlog']:
                selected_var_names = self.graph.nodes[child_node]['selected_var_names']

            if data_type == 'counts':
                x = relevant_cells[:, selected_var_names].X

            elif data_type == 'normlog':
                self.ensure_normlog()
                x = relevant_cells[:, selected_var_names].layers['normlog']

            else:
                raise Exception('Data type not currently supported.')

            if hasattr(x, 'todense'):
                x = x.todense()

            x = z_transform_properties(x)
            # the second output node (at index 1) is defined to represent a positive prediction for the node in question
            if activations_positive is None:
                activations_positive = self.graph.nodes[child_node]['local_classifier'].predict(x)[:, 1]

            else:
                # each row represents all activations (across cells) for a single possible label
                # each column represents the activations of the output node representing a positive result for all possible labels
                activations_positive = np.vstack((
                    activations_positive,
                    self.graph.nodes[child_node]['local_classifier'].predict(x)[:, 1]))

        if len(predicted_nodes) == 0:
            return

        if get_activations:
            return list(relevant_cells.obs_names), activations_positive

        if len(activations_positive.shape) > 1:
            y_int = np.argmax(activations_positive, axis = 0)

        else:
            y_int = np.array([0 for i in activations_positive])

        if self.prob_based_stopping:
            if 'threshold' in self.graph.nodes[node]:
                threshold = self.graph.nodes[node]['threshold']

            else:
                threshold = self.threshold

            if len(activations_positive.shape) > 1:
                len_set = np.sum(activations_positive >= threshold, axis=0)

            else:
                len_set = np.zeros((activations_positive.shape[0]))
                len_set[activations_positive >= threshold] = 1

            y = np.array([predicted_nodes[i] for i in y_int])
            y[len_set != 1] = 'stopped'

        else:
            y = [predicted_nodes[i] for i in y_int]

        obs_key = self.get_children_obs_key(node)
        self.set_predictions(obs_key, list(relevant_cells.obs_names), y)
        if node not in self.predictions.keys():
            self.predictions[node] = {}

        timestamp = str(time()).replace('.', '_')
        self.predictions[node][timestamp] = {
            'barcodes': list(relevant_cells.obs_names),
            'node': node,
            'data_type': [self.graph.nodes[n]['local_classifier'].data_type for n in predicted_nodes],
            'type_classifier': [type(self.graph.nodes[n]['local_classifier']) for n in predicted_nodes],
            'var_names': selected_var_names,
        }

    def predict_all_child_nodes_CPN(self, node, initial_call=True):
        """"""

        # Can not train root node classifier as there are no negative cells
        if len(self.get_child_nodes(node)) != 0:
            self.predict_single_parent_node_CPN(node)

        for child_node in self.get_child_nodes(node):
            self.predict_all_child_nodes_CPN(child_node, initial_call=False)

        if initial_call:
            if "overall" not in self.predictions.keys():
                self.predictions['overall'] = {}

            timestamp = str(time()).replace('.', '_')
            self.predictions['overall'][timestamp] = {
                'current_node': node
            }

    def update_hierarchy_CPN(self, dict_of_cell_relations, root_node=None):
        """"""

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

            print(f'Transfered to {node}, local classifier {"transferred" if "local_classifier" in self.graph.nodes[node] else "not transferred"}')

        self.graph = new_graph