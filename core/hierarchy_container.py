import networkx as nx
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from classiFire.core.tools import flatten_dict, dict_depth, hierarchy_names_unique, \
    make_graph_from_edges, set_node_to_depth, set_node_to_scVI
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.celltypist import CellTypistWrapper

class HierarchyContainer():
    """Add explanation
    """

    def __init__(self, dict_of_cell_relations, obs_names):
        """Add explanation
        """

        self.dict_of_cell_relations = dict_of_cell_relations
        self.obs_names = obs_names

        # Basic validation
        self.ensure_depth_match()
        self.ensure_unique_nodes()

        self.all_nodes = flatten_dict(self.dict_of_cell_relations)
        self.node_to_depth = set_node_to_depth(self.dict_of_cell_relations)
        self.node_to_scVI = set_node_to_scVI(self.dict_of_cell_relations)
        self.make_classifier_graph()

    def ensure_depth_match(self):
        """Check if the annotations supplied in .obs under obs_names are sufficiently deep to work 
        with the hierarchy provided.
        """

        if not dict_depth(self.dict_of_cell_relations) == len(self.obs_names):
            raise Exception('obs_names must contain an annotation key for every level of the '\
                'hierarchy supplied in dict_of_cell_relations.')

    def ensure_unique_nodes(self):
        """Check if keys within the hierarchy are unique across all levels as that is a requirement
        for uniquely identifying graph nodes with networkx.
        """

        if not hierarchy_names_unique(self.dict_of_cell_relations):
            raise Exception('Names given in the hierarchy must be unique.')

    def make_classifier_graph(self):
        """Compute directed graph from a given dictionary of cell relationships."""

        self.graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, self.graph)

    def plot_hierarchy(self):
        """Plot hierarchical cell labels.
        """

        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos, with_labels=True, arrows=False)

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

    def ensure_existence_classifier(self, node, input_len, classifier=NeuralNetwork, **kwargs):
        """Ensure that for the specified node in the graph, a local classifier exists under the
        key 'local_classifier'.
        """
        
        output_len = len(list(self.graph.adj[node].keys()))
        define_classifier = False
        if not 'local_classifier' in self.graph.nodes[node].keys():
            define_classifier = True

        else:
            try:
                current_input_len = self.graph.nodes[node]['local_classifier'].n_input
                current_output_len = self.graph.nodes[node]['local_classifier'].n_output
                if current_input_len != input_len or current_output_len != output_len:
                    # At some point (once changing the hierarchy becomes a thing) this should 
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

    def train_single_node(self, node, x, y_int=None, y_onehot=None, y=None, type_classifier=None):
        """Add explanation.
        """

        self.graph.nodes[node]['local_classifier'].train(x=x, y_onehot=y_onehot, y=y, y_int=y_int)
        train_acc, train_con_mat = self.graph.nodes[node]['local_classifier'].validate(x=x, y_int=y_int, y=y)
        self.graph.nodes[node]['last_train_acc'] = train_acc
        self.graph.nodes[node]['last_train_con_mat'] = train_con_mat

    def get_child_nodes(self, node):
        return self.graph.adj[node].keys()

    def is_trained_at(self, node):
        return 'local_classifier' in self.graph.nodes[node].keys()

    def predict_single_node(self, node, x, type_classifier):
        """Add explanation.
        """

        if type_classifier == CellTypistWrapper:
            y_pred = self.graph.nodes[node]['local_classifier'].predict(x)

        else: # type_classifier == type(NeuralNetwork):
            y_pred_int = self.graph.nodes[node]['local_classifier'].predict(x)
            y_pred = self.graph.nodes[node]['label_encoder'].inverse_transform(y_pred_int)

        return y_pred

    def set_preferred_classifier(self, node, type_classifier):
        self.graph.nodes[node]['preferred_classifier'] = type_classifier

    def get_selected_var_names(self, node):
        if not 'selected_var_names' in self.graph.nodes[node].keys():
            return None

        else:
            return self.graph.nodes[node]['selected_var_names']

    def set_selected_var_names(self, node, selected_var_names):
        self.graph.nodes[node]['selected_var_names'] = selected_var_names