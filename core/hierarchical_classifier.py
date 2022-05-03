from classiFire.core.tools import z_transform_properties
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.celltypist import CellTypistWrapper
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from uncertainties import ufloat
from copy import deepcopy
import numpy as np

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
        save_path,
        n_dimensions_scVI=30):

        self.data_container = data_container
        self.hierarchy_container = hierarchy_container
        self.save_path = save_path
        self.n_dimensions_scVI = n_dimensions_scVI

    def get_training_data_scVI(
        self, 
        node,
        barcodes, 
        scVI_key, 
        obs_name_children):
        """Gets untransformed input and target data for the training of a local classifier, 
        z-transforms the input data (scVI dimensions in the case of NeuralNetwork), calls upon
        self.hierarchy_container to encode the cell type labels into onehot and integer format using
        the label encoder at the local node.

        Parameters
        ----------
        node
            Specifies which local classifier is currently being trained. This information is
            used to access the relevant label encoder in self.hierarchy_container.transform_y.
        barcodes
            Specifies which cells should be used for training, enabling the retrieval of input
            and target data for these cells only.
        scVI_key
            Specifies under which key in data_container.adata.obsm the relevant scVI dimensions
            are to be found.
        obs_name_children
            Specifies under which key in data_container.adata.obs the target label relevant at this node
            is saved for each cell.
        """
        
        x, y = self.data_container.get_x_y_untransformed_scVI(barcodes, scVI_key, obs_name_children)
        x = z_transform_properties(x)
        y_int, y_onehot = self.hierarchy_container.transform_y(node, y)

        return x, y_int, y_onehot, y

    def train_single_node(
        self, 
        node, 
        barcodes=None):
        """Trains the local classifier stored at node.

        Parameters
        ----------
        node
            Specifies which local classifier is to be trained. node='T' would result in training
            of the classifier further differentiating between T cells.
        barcodes
            Specifies which cells should be used for training.
        n_dimensions_scVI
            Specifies the number of dimensions to run scVI with and to use as input into the 
            classifier.
        """

        # TODO
        # Overwrite n_dimensions_scVI if classifier already exists
        n_dimensions_scVI = self.n_dimensions_scVI
        print(f'Training classifier at {node}.')
        if type(barcodes) != type(None):
            print(f'Subsetting to {len(barcodes)} cells based on node assignment and'\
            ' designation as training data.')

        type_classifier = self.hierarchy_container.ensure_existence_classifier(
            node, 
            n_dimensions_scVI,
            classifier=CellTypistWrapper)
        obs_name_children = self.hierarchy_container.get_children_obs_key(node)
        self.hierarchy_container.ensure_existence_label_encoder(node)
        if type(barcodes) == type(None):
            barcodes = self.data_container.adata.obs_names

        if type_classifier == NeuralNetwork:        
            # Choose the most cell-type specific scVI dimensions available.
            scVI_node = self.hierarchy_container.node_to_scVI[node]
            scVI_key = self.data_container.get_scVI_key(
                node=scVI_node, 
                n_dimensions=n_dimensions_scVI,
                barcodes=barcodes)
            x, y_int, y_onehot, y = self.get_training_data_scVI(node, barcodes, scVI_key, obs_name_children)
            self.hierarchy_container.train_single_node(node, x=x, y_int=y_int, y_onehot=y_onehot, y=y, type_classifier=type_classifier)

        elif type_classifier == CellTypistWrapper:
            # TODO
            # Currently overwrites the model with every training
            x, y = self.data_container.get_x_y_untransformed_normlog(barcodes, obs_name_children)
            self.hierarchy_container.train_single_node(node, x=x, y=y, type_classifier=type_classifier)

    def train_all_child_nodes(
        self,
        current_node,
        train_barcodes=None):
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
        barcodes=None):
        """Uses an existing classifier at node to assign one of the child labels to the cells
        specified by barcodes. The predictions are stored in self.data_container.adata.obs by calling
        self.data_container.set_predictions under f'{obs_key}_pred' where obs_key is the key under
        which the annotations as one of the child labels are stored. If train_barcodes is not None, only cells within
        that subset are used for training.

        Parameters
        ----------
        node
            Specifies which local classifier is to be used for prediction. node='T' would result in
            predicting affiliation to one the T cell sub-labels.
        barcodes
            Specifies which cells are to be labelled.
        n_dimensions_scVI
            Specifies the number of dimensions to run scVI with and to use as input into the 
            classifier.
        """

        # TODO
        # Overwrite n_dimensions_scVI if classifier already exists
        n_dimensions_scVI = self.n_dimensions_scVI
        print(f'Predicting cells at {node}.')
        if type(barcodes) == type(None):
            barcodes = self.data_container.adata.obs_names

        print(f'Making prediction for {len(barcodes)} cells based on node assignment and'\
            ' designation as prediction data.')
        if not self.hierarchy_container.is_trained_at(node):
            raise Exception(f'Must train local classifier for {node} before trying to predict cell'\
                ' types')


        type_classifier = self.hierarchy_container.ensure_existence_classifier(
            node, 
            n_dimensions_scVI,
            classifier=NeuralNetwork)
        if type_classifier == CellTypistWrapper:
            x = self.data_container.get_x_untransformed_normlog(barcodes)

        else: # type_classifier == type(NeuralNetwork):
            # Choose the most cell-type specific scVI dimensions available.
            scVI_node = self.hierarchy_container.node_to_scVI[node]
            scVI_key = self.data_container.get_scVI_key(
                node=scVI_node, 
                n_dimensions=n_dimensions_scVI,
                barcodes=barcodes)
            x = self.data_container.get_x_untransformed_scVI(barcodes, scVI_key)
            x = z_transform_properties(x)

        y_pred = self.hierarchy_container.predict_single_node(node, x, type_classifier=type_classifier)
        obs_key = self.hierarchy_container.get_children_obs_key(node)
        self.data_container.set_predictions(obs_key, barcodes, y_pred)

    def predict_all_child_nodes(
        self,
        current_node,
        current_barcodes=None,
        test_barcodes=None):
        """Starts at current_node, predicting cell label affiliation using its local classifier.
        Recursively predicts affiliation to cell type labels lower in the hierarchy, using as relevant
        cell subgroup those cells that were predicted to belong to the parent node label. If 
        test_barcodes is not None, only cells within that subset are used for prediction, enabling
        cross-validation.

        Parameters
        ----------
        current_node
            When initially calling the method this refers to the node in the hierarchy at which
            to start prediction. Keeps track of which node is currently being predicted for over 
            the course of the recursion.
        test_barcodes
            Barcodes (adata.obs_names) of those cells that are available for prediction. Necessary 
            to enable separation of training and test data for cross-validation and make predictions
            only for those cells which the classifier has not yet seen.
        """

        if type(current_barcodes) == type(None):
            current_barcodes = self.data_container.adata.obs_names

        if type(test_barcodes) != type(None):
            current_barcodes = [b for b in current_barcodes if b in test_barcodes]

        self.predict_single_node(current_node, current_barcodes)
        obs_key = self.hierarchy_container.get_children_obs_key(current_node)
        for child_node in self.hierarchy_container.get_child_nodes(current_node):
            if len(self.hierarchy_container.get_child_nodes(child_node)) == 0:
                continue

            child_node_barcodes = self.data_container.get_predicted_barcodes(
                obs_key, 
                child_node,
                predicted_from=test_barcodes)
            self.predict_all_child_nodes(child_node, child_node_barcodes)

    def train_child_nodes_with_validation(
        self, 
        starting_node,
        y_obs=None,
        barcodes=None,
        k=None,
        test_size=0.25,
        isolate_test_network=True):
        """Implements automatic hierarchical/recursive classification along the hierarchy
        in combination with a test/train split (simple or k-fold CV if k is not None).

        Parameters
        ----------
        starting_node
            When initially calling the method this refers to the node in the hierarchy at which
            to start training and later on testing. Keeps track of which node is currently being
            trained or tested over the course of the recursion.
        y_obs
            Specifies which obs key/level of labels is considered for overall accuracy determination
            and construction of the confusion matrix. If y_obs == None, is set to the last level
            in the hierarchy.
        barcodes
            Barcodes of cells to consider for training and testing.
        k
            Number of test splits for k-fold CV. If k == None, a simpel train_test_split is used.
        test_size
            Percentage of barcodes to withhold for testing in the event a simple train_test_split
            is used.
        isolate_test_network
            If true, the classification models stored in the hierarchy are left untouched by this 
            method, restoring them to their original state after each train/test sequence. If false,
            the models would have previously seen any test data at least once after the first
            train/test sequence of k-fold CV, rendering all results basically useless.
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
            con_mats = []
            accs = []
            for barcodes_train_idx, barcodes_test_idx in skf.split(barcodes, y):
                if isolate_test_network:
                    self.hierarchy_container_copy = deepcopy(self.hierarchy_container)

                barcodes_train = barcodes[barcodes_train_idx]
                barcodes_test = barcodes[barcodes_test_idx]
                self.train_all_child_nodes(starting_node, barcodes_train)
                self.predict_all_child_nodes(starting_node, test_barcodes=barcodes_test)
                acc, con_mat, possible_labels = self.data_container.get_total_accuracy(y_obs, test_barcodes=barcodes_test)
                con_mats.append(con_mat)
                accs.append(acc)
                if isolate_test_network:
                    self.hierarchy_container = deepcopy(self.hierarchy_container_copy)

            averaged_con_mat = np.sum(con_mats, axis=0) / np.sum(np.sum(con_mats, axis=0), axis=1)
            test_score_mean = ufloat(np.mean(accs), np.std(accs))
            print('Average con mat')
            disp = ConfusionMatrixDisplay(confusion_matrix=averaged_con_mat, display_labels=possible_labels)
            disp.plot()
            print(f'Test accuracy was {test_score_mean}')

    def set_classifier_type(self, node, preferred_classifier):
        if type(node) == list:
            for n in node:
                self.hierarchy_container.set_preferred_classifier(n, preferred_classifier)

        else:
            self.hierarchy_container.set_preferred_classifier(node, preferred_classifier)