from classiFire.core.tools import z_transform_properties
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.celltypist import CellTypistWrapper
from classiFire.core.models.logreg import LogRegWrapper
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from uncertainties import ufloat
from copy import deepcopy
from imblearn.over_sampling import SMOTE
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
        n_dimensions_scVI=30,
        prob_based_stopping = False,
        use_scVI=False,
        use_norm_X=True,
        use_feature_selection=True,
        n_top_genes_per_class=300,
        sampling_method=SMOTE,
        sampling_strategy='auto'):

        self.data_container = data_container
        self.hierarchy_container = hierarchy_container
        self.save_path = save_path
        self.n_dimensions_scVI = n_dimensions_scVI
        self.prob_based_stopping = prob_based_stopping
        self.use_scVI = use_scVI
        self.use_norm_X = use_norm_X
        self.use_feature_selection = use_feature_selection
        self.n_top_genes_per_class = n_top_genes_per_class
        if type(sampling_method) != type(None):
            self.data_container.init_resampling(sampling_method, sampling_strategy)

    def get_selected_var_names(self, node, barcodes, obs_name_children=None):
        if self.use_scVI == True:
            return None

        var_names = self.hierarchy_container.get_selected_var_names(node)
        if type(var_names) == type(None) and self.use_feature_selection == True:
            var_names = self.data_container.get_top_genes(
                barcodes, 
                obs_name_children, 
                self.n_top_genes_per_class)
            self.hierarchy_container.set_selected_var_names(node, var_names)

        return var_names

    def get_training_data(
        self, 
        node,
        barcodes,
        obs_name_children,
        scVI_key=None,
        use_counts=False):
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

        var_names = self.get_selected_var_names(node, barcodes, obs_name_children)
        if self.use_scVI == True:
            data = 'scVI'

        elif self.use_norm_X == True:
            data = 'normlog'

        else:
            data = 'counts'

        if use_counts:
            data = 'counts'

        return_adata = False
        if self.hierarchy_container.get_preferred_classifier(node) == CellTypistWrapper:
            return_adata = True

        x, y = self.data_container.get_x_y_untransformed(
            barcodes=barcodes, 
            obs_name_children=obs_name_children, 
            data=data, 
            var_names=var_names, 
            scVI_key=scVI_key, 
            return_adata=return_adata)

        if return_adata == False:
            if self.hierarchy_container.get_preferred_classifier(node) != LogRegWrapper:
                x = z_transform_properties(x)

            else:
                self.hierarchy_container.fit_chi2_feature_selecter(node, x, y)
                x = self.hierarchy_container.graph.nodes[node]['chi2_feature_selecter'].transform(x)

            y_int, y_onehot = self.hierarchy_container.transform_y(node, y)

            return x, y, y_int, y_onehot

        else:
            return x, y, None, None

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
        """

        print(f'Training classifier at {node}.')
        if type(barcodes) != type(None):
            print(f'Subsetting to {len(barcodes)} cells based on node assignment and'\
            ' designation as training data.')

        else:
            barcodes = self.data_container.adata.obs_names

        obs_name_children = self.hierarchy_container.get_children_obs_key(node)
        var_names = self.get_selected_var_names(node, barcodes, obs_name_children)
        self.hierarchy_container.ensure_existence_label_encoder(node)
        self.hierarchy_container.set_chi2_feature_selecter(node)
        type_classifier = self.hierarchy_container.get_preferred_classifier(node)
        if type(type_classifier) == type(None):
            type_classifier = NeuralNetwork

        scVI_key = None
        use_counts = False
        if type_classifier == NeuralNetwork:
            if self.use_scVI == True:
                scVI_node = self.hierarchy_container.node_to_scVI[node]
                scVI_key = self.data_container.get_scVI_key(
                    node=scVI_node, 
                    n_dimensions=self.n_dimensions_scVI,
                    barcodes=barcodes)

        elif type_classifier == CellTypistWrapper:
            pass

        elif type_classifier == LogRegWrapper:
            use_counts = True

        else:
            raise Exception('The local classifier type you have chosen is not currently implemented.')

        x, y, y_int, y_onehot = self.get_training_data(node, barcodes, obs_name_children, scVI_key=scVI_key, use_counts=use_counts)
        if self.use_scVI:
            input_n_classifier = self.n_dimensions_scVI

        elif type(var_names) == type(None):
            input_n_classifier = len(self.data_container.adata.var_names)

        else:
            input_n_classifier = len(var_names)

        self.hierarchy_container.ensure_existence_classifier(
            node, 
            input_n_classifier,
            classifier=type_classifier)
        self.hierarchy_container.train_single_node(node, x=x, y_int=y_int, y_onehot=y_onehot, y=y)

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
        """

        print(f'Predicting cells at {node}.')
        if type(barcodes) == type(None):
            barcodes = self.data_container.adata.obs_names

        print(f'Making prediction for {len(barcodes)} cells based on node assignment and'\
            ' designation as prediction data.')
        if not self.hierarchy_container.is_trained_at(node):
            raise Exception(f'Must train local classifier for {node} before trying to predict cell'\
                ' types')

        scVI_key = None
        if self.use_scVI == True:
            # Choose the most cell-type specific scVI dimensions available.
            scVI_node = self.hierarchy_container.node_to_scVI[node]
            scVI_key = self.data_container.get_scVI_key(
                node=scVI_node, 
                n_dimensions=self.n_dimensions_scVI,
                barcodes=barcodes)

        var_names = self.get_selected_var_names(node, barcodes)
        if self.use_scVI == True:
            data = 'scVI'

        elif self.use_norm_X == True:
            data = 'normlog'

        else:
            data = 'counts'

        print(f'Predicting with {len(var_names) if type(var_names) != type(None) else "all available"} genes')
        return_adata = False
        type_classifier = self.hierarchy_container.get_preferred_classifier(node)
        if type_classifier == CellTypistWrapper:
            return_adata = True

        elif type_classifier == LogRegWrapper:
            data = 'counts'

        x = self.data_container.get_x_untransformed(barcodes, data=data, var_names=var_names, scVI_key=scVI_key, return_adata=return_adata)
        if return_adata == False and not type_classifier == LogRegWrapper:
            x = z_transform_properties(x)

        elif type_classifier == LogRegWrapper:
            x = self.hierarchy_container.graph.nodes[node]['chi2_feature_selecter'].transform(x)

        if not self.prob_based_stopping:
            y_pred = self.hierarchy_container.predict_single_node(node, x, type_classifier=type_classifier)
            obs_key = self.hierarchy_container.get_children_obs_key(node)
            self.data_container.set_predictions(obs_key, barcodes, y_pred)

        #%--------------------------------------------------------------------------------------------------------------------------------------------%#       
        #belongs somewhere in the prediction methds, not sure where yet because of test/training problem
        #%--------------------------------------------------------------------------------------------------------------------------------------------%#

        elif self.prob_based_stopping:
            y_pred = self.hierarchy_container.predict_single_node_proba(node, x, type_classifier=type_classifier)
            #child_obs_key says at which hierarchy level the predictions have to be saved
            child_obs_key = self.hierarchy_container.get_children_obs_key(node) 
            parent_obs_key = self.hierarchy_container.get_parent_obs_key(node)
            self.data_container.set_prob_based_predictions(node, child_obs_key, parent_obs_key, barcodes, y_pred, fitted_label_encoder=self.hierarchy_container.graph.nodes[node]['label_encoder'])

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

        self.predict_single_node(current_node, barcodes=current_barcodes)
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

            barcodes_train, barcodes_test = train_test_split(barcodes, test_size=test_size, stratify = y)
            self.train_all_child_nodes(starting_node, barcodes_train)
            self.predict_all_child_nodes(starting_node, test_barcodes=barcodes_test)
            self.data_container.get_total_accuracy(y_obs, test_barcodes=barcodes_test)
            #integrate preliminary hierarchical confusion matrix
            # self.data_container.get_hierarchical_accuracy(test_barcodes=barcodes_test, level_obs_keys=self.hierarchy_container.obs_names, all_labels = self.hierarchy_container.all_nodes, overview_obs_key = 'Level_2' )
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
                # acc, con_mat, possible_labels, con_mat_overview, possible_labels_overview = self.data_container.get_hierarchical_accuracy(test_barcodes=barcodes_test, level_obs_keys=self.hierarchy_container.obs_names, all_labels=self.hierarchy_container.all_nodes, overview_obs_key = 'Level_2')
                con_mats.append(con_mat)
                # con_mats.append(con_mat_overview)
                accs.append(acc)
                if isolate_test_network:
                    self.hierarchy_container = deepcopy(self.hierarchy_container_copy)

            averaged_con_mat = np.sum(con_mats, axis=0) / np.sum(np.sum(con_mats, axis=0), axis=1)
            test_score_mean = ufloat(np.mean(accs), np.std(accs))
            print('Average con mat')
            disp = ConfusionMatrixDisplay(confusion_matrix=averaged_con_mat, display_labels=possible_labels)
            disp.plot(xticks_rotation='vertical')
            print(f'Test accuracy was {test_score_mean}')

    def set_classifier_type(self, node, preferred_classifier):
        if type(node) == list:
            for n in node:
                self.hierarchy_container.set_preferred_classifier(n, preferred_classifier)

        else:
            self.hierarchy_container.set_preferred_classifier(node, preferred_classifier)