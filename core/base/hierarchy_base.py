import networkx as nx
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from sklearn.preprocessing import LabelEncoder
from classiFire.core.tools import flatten_dict, dict_depth, hierarchy_names_unique, \
    make_graph_from_edges, set_node_to_depth, set_node_to_scVI
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.celltypist import CellTypistWrapper
from classiFire.core.models.logreg import LogRegWrapper
from classiFire.core.models.single_assignment import SingleAssignment
from sklearn.feature_selection import SelectKBest, chi2

class HierarchyBase():
    """Add explanation
    """

    def set_cell_relations(self, dict_of_cell_relations, obs_names):
        """Once set, cell relations can only be changed one node at a time, using supplied methods,
        not by simply calling defining new cell relations
        """

        if type(self.dict_of_cell_relations) != type(None) and type(self.obs_names) != type(None):
            raise Exception('Cannot redefine cell relations after initialization.')

        self.dict_of_cell_relations = dict_of_cell_relations
        self.obs_names = obs_names
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

    def set_chi2_feature_selecter(self, node, number_features=50):
        """save chi2 feature selecter in one node, train only once""" 

        # To do: is this really only trained once or is that something that needs to be implemented???

        self.graph.nodes[node]['chi2_feature_selecter'] = SelectKBest(chi2, k = number_features)
        self.graph.nodes[node]['chi2_feature_selecter_trained'] = False

    def fit_chi2_feature_selecter(self, node, x_feature_fit, y_feature_fit):
        """fit chi2 feature selecter once, i.e. only with one trainings dataset"""

        # To do: see above? need to be sure that features are not selected again for every new training run

        if self.graph.nodes[node]['chi2_feature_selecter_trained'] == False:
            self.graph.nodes[node]['chi2_feature_selecter'].fit(x_feature_fit, y_feature_fit)
            self.graph.nodes[node]['chi2_feature_selecter_trained'] = True

        else: 
            print('Chi2 Feature selecter already trained, using trained selecter!')

    def ensure_existence_classifier(self, node, input_len, classifier=NeuralNetwork, **kwargs):
        """Ensure that for the specified node in the graph, a local classifier exists under the
        key 'local_classifier'.
        """
        
        output_len = len(list(self.graph.adj[node].keys()))
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

        if hasattr(self, 'default_input_data') and self.default_input_data in self.graph.nodes[node]['local_classifier'].possible_data_types:
            self.graph.nodes[node]['local_classifier'].data_type = self.default_input_data

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

        # To do: make argument on instantiation
        threshold = 0.9

        # print(f'type_classifier from predict_.._proba: {type_classifier}')
        type_classifier = self.graph.nodes[node]['local_classifier']
        if type_classifier == NeuralNetwork:
            y_pred_proba = self.graph.nodes[node]['local_classifier'].predict_proba(x)
            #y_pred_proba array_like with length of predictable classes, entries of form x element [0,1]
            #with sum(y_pred) = 1 along axis 1 (for one cell)

            #print(f'y_pred_proba: {y_pred_proba[:10]}')
            #print(f'y_pred_proba.shape: {y_pred_proba.shape}')

            #test if probability for one class is larger than threshold 
            largest_idx = np.argmax(y_pred_proba, axis = -1)

            #print(f'largest_idx: {largest_idx[:10]}')

            #y_pred is real prediction vector, with possible nans (else case)!
            y_pred = []
            for cell_idx, label_idx in enumerate(largest_idx):
                if y_pred_proba[cell_idx][label_idx] > threshold:
                    #in this case: set prediction and move on to next classifier
                    y_pred.append(label_idx) #label_idx = class per definition
                else: 
                    #otherwise, set no new prediction, predition from superior node shall be set as 
                    #the last prediction
                    y_pred.append(np.nan)

        else:
            raise ValueError('Not yet supported probability classification classifier type')

        return y_pred

    def set_preferred_classifier(self, node, type_classifier):
        self.graph.nodes[node]['preferred_classifier'] = type_classifier

    def get_selected_var_names(self, node, barcodes, obs_name_children=None, data_type='normlog'):
        if data_type == 'scVI':
            return None

        if not 'selected_var_names' in self.graph.nodes[node].keys():
            var_names = None

        else:
            var_names = self.graph.nodes[node]['selected_var_names']

        if type(var_names) == type(None) and self.use_feature_selection == True:
            var_names = self.get_top_genes(
                barcodes, 
                obs_name_children, 
                self.n_top_genes_per_class)
            self.set_selected_var_names(node, var_names)

        return var_names

    def set_selected_var_names(self, node, selected_var_names):
        self.graph.nodes[node]['selected_var_names'] = selected_var_names

    def get_preferred_classifier(self, node):
        if 'preferred_classifier' in self.graph.nodes[node].keys():
            return self.graph.nodes[node]['preferred_classifier']

        else:
            return None

    def get_leaf_nodes(self):
        return [
            x for x in self.graph.nodes() \
            if self.graph.out_degree(x) == 0 \
            and self.grpah.in_degree(x) == 1
        ]