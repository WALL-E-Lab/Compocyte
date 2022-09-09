import networkx as nx
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from classiFire.core.tools import flatten_dict, dict_depth, hierarchy_names_unique, \
    make_graph_from_edges, set_node_to_depth, delete_dict_entries
from classiFire.core.models.dense import DenseKeras
from classiFire.core.models.dense_torch import DenseTorch
from classiFire.core.models.log_reg import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from copy import deepcopy

class HierarchyBase():
    """Add explanation
    """

    def set_cell_relations(self, root_node, dict_of_cell_relations, obs_names):
        """Once set, cell relations can only be changed one node at a time, using supplied methods,
        not by simply calling defining new cell relations
        """

        if self.root_node is not None and self.dict_of_cell_relations is not None and self.obs_names is not None:
            raise Exception('To redefine cell relations after initialization, call update_hierarchy.')

        dict_of_cell_relations_with_classifiers = deepcopy(dict_of_cell_relations)
        dict_of_cell_relations, contains_classifier = delete_dict_entries(dict_of_cell_relations, 'classifier')
        self.root_node = root_node
        self.ensure_depth_match(dict_of_cell_relations, obs_names)
        self.ensure_unique_nodes(dict_of_cell_relations)
        self.dict_of_cell_relations = dict_of_cell_relations
        self.obs_names = obs_names
        self.all_nodes = flatten_dict(self.dict_of_cell_relations)
        self.node_to_depth = set_node_to_depth(self.dict_of_cell_relations)
        self.make_classifier_graph()        
        if contains_classifier:
            self.import_classifiers(dict_of_cell_relations_with_classifiers)

    def ensure_depth_match(self, dict_of_cell_relations, obs_names):
        """Check if the annotations supplied in .obs under obs_names are sufficiently deep to work 
        with the hierarchy provided.
        """

        if not dict_depth(dict_of_cell_relations) == len(obs_names):
            raise Exception('obs_names must contain an annotation key for every level of the '\
                'hierarchy supplied in dict_of_cell_relations.')

    def ensure_unique_nodes(self, dict_of_cell_relations):
        """Check if keys within the hierarchy are unique across all levels as that is a requirement
        for uniquely identifying graph nodes with networkx.
        """

        if not hierarchy_names_unique(dict_of_cell_relations):
            raise Exception('Names given in the hierarchy must be unique.')

    def make_classifier_graph(self):
        """Compute directed graph from a given dictionary of cell relationships."""

        self.graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, self.graph)

    def plot_hierarchy(self):
        """Plot hierarchical cell labels.
        """

        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='twopi')
        nx.draw(self.graph, pos, with_labels=True, arrows=True, ax=ax)

    def get_children_obs_key(self, parent_node):
        """Get the obs key under which labels for the following level in the hierarchy are saved.
        E. g. if you get_children_obs_key for T cells, it will return the obs key for alpha beta
        T cell labels and so on.
        """

        depth_parent = self.node_to_depth[parent_node]
        children_obs_key = self.obs_names[depth_parent + 1]

        return children_obs_key

    def get_parent_obs_key(self, parent_node):
        """Get the obs key under which labels for the current level in the hierarchy are saved.
        E. g. if you get_parent_obs_key for T cells, it will return the obs key in which true
        T cells are labelled as such.
        """

        depth_parent = self.node_to_depth[parent_node]

        return self.obs_names[depth_parent]

    def set_f_classif_feature_selecter(self, node, number_features=50):
        """save f_classif feature selecter in one node, train only once""" 

        # To do: is this really only trained once or is that something that needs to be implemented???

        self.graph.nodes[node]['f_classif_feature_selecter'] = SelectKBest(f_classif, k = number_features)
        self.graph.nodes[node]['f_classif_feature_selecter_trained'] = False

    def fit_f_classif_feature_selecter(self, node, x_feature_fit, y_feature_fit):
        """fit f_classif feature selecter once, i.e. only with one trainings dataset"""

        # To do: see above? need to be sure that features are not selected again for every new training run

        if self.graph.nodes[node]['f_classif_feature_selecter_trained'] == False:
            self.graph.nodes[node]['f_classif_feature_selecter'].fit(x_feature_fit, y_feature_fit)
            self.graph.nodes[node]['f_classif_feature_selecter_trained'] = True

        else: 
            print('f_classif Feature selecter already trained, using trained selecter!')

    def ensure_existence_OVR_classifier(self, node, n_input, type_classifier, data_type, **kwargs):
        print(f'Trying creating OVR at {node}')
        if 'local_classifier' in self.graph.nodes[node].keys():
            return

        self.graph.nodes[node]['local_classifier'] = type_classifier(n_input=n_input, n_output=2, **kwargs)
        self.graph.nodes[node]['local_classifier'].set_data_type(data_type)
        print(f'OVR classifier set up as {type_classifier} with {data_type} data at {node}.')

    def ensure_existence_classifier(self, node, input_len, classifier=DenseTorch, is_CPN=False, n_output=None, **kwargs):
        """Ensure that for the specified node in the graph, a local classifier exists under the
        key 'local_classifier'.
        """
        
        if n_output is None:
            output_len = len(list(self.graph.adj[node].keys()))

        else:
            output_len = n_output

        define_classifier = False
        if not 'local_classifier' in self.graph.nodes[node].keys() or type(self.graph.nodes[node]['local_classifier']) == SingleAssignment:
            define_classifier = True

        else:
            try:
                current_input_len = self.graph.nodes[node]['local_classifier'].n_input
                current_output_len = self.graph.nodes[node]['local_classifier'].n_output
                if current_input_len != input_len or current_output_len != output_len:
                    # To do: At some point (once changing the hierarchy becomes a thing) this should 
                    # change the layer structure, rather than overwriting it all together
                    define_classifier = True

            except AttributeError:
                print('There has either been an issue setting up input and output length of the local '\
                    'classifier or you\'re using a model that does not rely on these arguments.')

        if define_classifier:
            if 'preferred_classifier' in self.graph.nodes[node].keys():
                self.graph.nodes[node]['local_classifier'] = self.graph.nodes[node]['preferred_classifier'](n_input=input_len, n_output=output_len, **kwargs)

            else:
                self.graph.nodes[node]['local_classifier'] = classifier(n_input=input_len, n_output=output_len, **kwargs)

        if hasattr(self, 'default_input_data') and self.default_input_data in self.graph.nodes[node]['local_classifier'].possible_data_types or self.default_input_data in self.adata.obsm:
            self.graph.nodes[node]['local_classifier'].set_data_type(self.default_input_data)

        print(f'Data type for {node} set to {self.graph.nodes[node]["local_classifier"].data_type}')
        return type(self.graph.nodes[node]['local_classifier'])

    def ensure_existence_label_encoder(self, node):
        """Add explanation.
        """

        if not 'label_encoder' in self.graph.nodes[node].keys():
            label_encoder = LabelEncoder()
            children_labels = list(self.graph.adj[node].keys())
            label_encoder.fit(np.array(children_labels))
            self.graph.nodes[node]['label_encoder'] = label_encoder

    def transform_y(self, node, y):
        """Add explanation.
        """

        num_classes = len(list(self.graph.adj[node].keys()))
        y_int = self.graph.nodes[node]['label_encoder'].transform(y)
        y_onehot = keras.utils.to_categorical(y_int, num_classes=num_classes)

        return y_int, y_onehot

    def get_child_nodes(self, node):
        return self.graph.adj[node].keys()

    def is_trained_at(self, node):
        return 'local_classifier' in self.graph.nodes[node].keys()

    def predict_single_node_proba(self, node, x):
        """Predict output and fit downstream analysis based on a probability threshold (default = 90%)"""
        # print(f'type_classifier from predict_.._proba: {type_classifier}')
        type_classifier = type(self.graph.nodes[node]['local_classifier'])
        if type_classifier in [DenseKeras, DenseTorch, LogisticRegression]:
            y_pred_proba = self.graph.nodes[node]['local_classifier'].predict(x)
            #y_pred_proba array_like with length of predictable classes, entries of form x element [0,1]
            #with sum(y_pred) = 1 along axis 1 (for one cell)

            #print(f'y_pred_proba: {y_pred_proba[:10]}')
            #print(f'y_pred_proba.shape: {y_pred_proba.shape}')

            #test if probability for one class is larger than threshold 
            largest_idx = np.argmax(y_pred_proba, axis = -1)

            #print(f'largest_idx: {largest_idx[:10]}')
            if 'threshold' in self.graph.nodes[node]:
                threshold = self.graph.nodes[node]['threshold']

            else:
                threshold = self.threshold

            len_set = np.sum(y_pred_proba >= threshold, axis=1)
            largest_idx[len_set != 1] = np.nan

        else:
            raise ValueError('Not yet supported probability classification classifier type')

        return largest_idx

    def set_preferred_classifier(self, node, type_classifier):
        self.graph.nodes[node]['preferred_classifier'] = type_classifier

    def get_preferred_classifier(self, node):
        if 'preferred_classifier' in self.graph.nodes[node].keys():
            return self.graph.nodes[node]['preferred_classifier']

        else:
            return None

    def get_leaf_nodes(self):
        return [
            x for x in self.graph.nodes() \
            if self.graph.out_degree(x) == 0 \
            and self.graph.in_degree(x) == 1
        ]

    def get_parent_node(self, node, graph=None):
        if graph is None:
            graph = self.graph

        edges = np.array(graph.edges)
        # In a directed graph there should only be one edge leading TO any given node
        idx_child_node_edges = np.where(edges[:, 1] == node)
        parent_node = edges[idx_child_node_edges][0, 0]

        return parent_node

    def update_hierarchy(self, dict_of_cell_relations, root_node=None, overwrite=False):
        dict_of_cell_relations_with_classifiers = deepcopy(dict_of_cell_relations)
        dict_of_cell_relations, contains_classifier = delete_dict_entries(dict_of_cell_relations, 'classifier')
        if self.classification_mode == 'CPN':
            self.update_hierarchy_CPN(dict_of_cell_relations, root_node=root_node)

        elif self.classification_mode == 'CPPN':
            self.update_hierarchy_CPPN(dict_of_cell_relations, root_node=root_node)

        else:
            raise Exception('Classification mode unknown.')

        if contains_classifier:
            self.import_classifiers(dict_of_cell_relations_with_classifiers, overwrite=overwrite)