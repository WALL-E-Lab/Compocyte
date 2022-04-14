from classiFire.core.tools import z_transform_properties
from classiFire.core.neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split, StratifiedKFold
from copy import deepcopy

class HierarchicalClassifier():
    """This class coordinates the passing of information between the cell label hierarchy (in
    HierarchyContainer), the sequencing and cell-level data (in SequencingDataContainer) and the
    local classifiers making classification decisions for single nodes in the cell label hierarchy.

    Parameters
    ----------
    data_container
        Object of type classiFire.core.sequencing_data_container.SequencingDataContainer.
    hierarchy_container
        Object of type classiFire.core.hierarchy_container.HierarchyContainer.
    save_path
        Path to save model states and other information processed by this class.
    """ 

    def __init__(
        self, 
        data_container, 
        hierarchy_container, 
        save_path):

        self.data_container = data_container
        self.hierarchy_container = hierarchy_container
        self.save_path = save_path

    def get_training_data(
        self, 
        node,
        barcodes, 
        scVI_key, 
        obs_name_children):
        """Add explanation
        """
        
        x, y = self.data_container.get_x_y_untransformed(barcodes, scVI_key, obs_name_children)
        x = z_transform_properties(x)
        y_int, y_onehot = self.hierarchy_container.transform_y(node, y)

        return x, y_int, y_onehot

    def train_single_node(
        self, 
        node, 
        barcodes=None, 
        n_dimensions_scVI=10):
        """Add explanation
        """

        print(f'Training classifier at {node}.')
        if type(barcodes) != type(None):
            print(f'Subsetting to {len(barcodes)} cells based on node assignment and'\
            ' designation as training data.')

        # Choose the most cell-type specific scVI dimensions available.
        scVI_node = self.hierarchy_container.node_to_scVI[node]
        scVI_key = self.data_container.get_scVI_key(
            node=scVI_node, 
            n_dimensions=n_dimensions_scVI,
            barcodes=barcodes)
        self.hierarchy_container.ensure_existence_classifier(
            node, 
            n_dimensions_scVI,
            classifier=NeuralNetwork)
        self.hierarchy_container.ensure_existence_label_encoder(node)
        if type(barcodes) == type(None):
            barcodes = self.data_container.adata.obs_names

        obs_name_children = self.hierarchy_container.get_children_obs_key(node)
        x, y_int, y_onehot = self.get_training_data(node, barcodes, scVI_key, obs_name_children)
        self.hierarchy_container.train_single_node(node, x, y_int, y_onehot)

    def train_all_child_nodes(
        self,
        current_node,
        train_barcodes=None):
        """Add explanation
        """

        obs_name_parent = self.hierarchy_container.get_parent_obs_key(current_node)
        true_barcodes = self.data_container.get_true_barcodes(
            obs_name_parent, 
            current_node, 
            true_from=train_barcodes)
        self.train_single_node(current_node, true_barcodes)
        for child_node in self.hierarchy_container.get_child_nodes(current_node):
            if len(self.hierarchy_container.get_child_nodes(child_node)) == 0:
                continue

            self.train_all_child_nodes(child_node, train_barcodes=train_barcodes)

    def predict_single_node(
        self,
        node,
        barcodes=None,
        n_dimensions_scVI=10,
        test_barcodes=None):
        """Add explanation.
        """

        print(f'Predicting cells at {node}.')
        if type(barcodes) == type(None):
            barcodes = self.data_container.adata.obs_names

        if type(test_barcodes) != type(None):
            barcodes = [b for b in barcodes if b in test_barcodes]

        print(f'Making prediction for {len(barcodes)} cells based on node assignment and'\
            ' designation as prediction data.')
        # Choose the most cell-type specific scVI dimensions available.
        scVI_node = self.hierarchy_container.node_to_scVI[node]
        scVI_key = self.data_container.get_scVI_key(
            node=scVI_node, 
            n_dimensions=n_dimensions_scVI,
            barcodes=barcodes)
        if not self.hierarchy_container.is_trained_at(node):
            raise Exception(f'Must train local classifier for {node} before trying to predict cell'\
                ' types')

        x = self.data_container.get_x_untransformed(barcodes, scVI_key)
        x = z_transform_properties(x)
        y_pred = self.hierarchy_container.predict_single_node(node, x)
        obs_key = self.hierarchy_container.get_children_obs_key(node)
        self.data_container.set_predictions(obs_key, barcodes, y_pred)

    def predict_all_child_nodes(
        self,
        current_node,
        current_barcodes=None,
        test_barcodes=None):
        """Add explanation.
        """

        self.predict_single_node(current_node, current_barcodes, test_barcodes=test_barcodes)
        obs_key = self.hierarchy_container.get_children_obs_key(current_node)
        for child_node in self.hierarchy_container.get_child_nodes(current_node):
            if len(self.hierarchy_container.get_child_nodes(child_node)) == 0:
                continue

            child_node_barcodes = self.data_container.get_predicted_barcodes(
                obs_key, 
                child_node)
            self.predict_all_child_nodes(child_node, child_node_barcodes, test_barcodes)

    def train_child_nodes_with_validation(
        self, 
        starting_node,
        y_obs=None,
        barcodes=None,
        k=None,
        test_size=0.25,
        isolate_test_network=True):
        """Add explanation.
        """

        if type(y_obs) == type(None):
            y_obs = self.hierarchy_container.obs_names[-1]
        
        y = self.data_container.adata.obs[y_obs]
        if type(barcodes) == type(None):
            barcodes = self.data_container.adata.obs_names

        if type(k) == type(None):
            if isolate_test_network:
                self.hierarchy_container_copy = deepcopy(self.hierarchy_container)

            barcodes_train, barcodes_test = train_test_split(barcodes, test_size=test_size)
            self.train_all_child_nodes(starting_node, barcodes_train)
            self.predict_all_child_nodes(starting_node, test_barcodes=barcodes_test)
            self.data_container.get_total_accuracy(y_obs, test_barcodes=barcodes_test)
            if isolate_test_network:
                self.hierarchy_container = deepcopy(self.hierarchy_container_copy)

        else:
            skf = StratifiedKFold(n_splits=k)
            for barcodes_train_idx, barcodes_test_idx in skf.split(barcodes, y):
                if isolate_test_network:
                    self.hierarchy_container_copy = deepcopy(self.hierarchy_container)

                barcodes_train = barcodes[barcodes_train_idx]
                barcodes_test = barcodes[barcodes_test_idx]
                self.train_all_child_nodes(starting_node, barcodes_train)
                self.predict_all_child_nodes(starting_node, test_barcodes=barcodes_test)
                self.data_container.get_total_accuracy(y_obs, test_barcodes=barcodes_test)
                if isolate_test_network:
                    self.hierarchy_container = deepcopy(self.hierarchy_container_copy)