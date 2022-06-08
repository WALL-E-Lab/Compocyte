from classiFire.core.base.data_base import DataBase
from classiFire.core.base.hierarchy_base import HierarchyBase
from classiFire.core.tools import z_transform_properties, flatten_dict, set_node_to_depth, set_node_to_scVI
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.celltypist import CellTypistWrapper
from classiFire.core.models.logreg import LogRegWrapper
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from uncertainties import ufloat
from copy import deepcopy
from imblearn.over_sampling import SMOTE
import numpy as np

class HierarchicalClassifier(DataBase, HierarchyBase):
    """Add explanation
    """

    def __init__(
        self,
        save_path,
        adata = None,
        dict_of_cell_relations=None, 
        obs_names=None,
        n_dimensions_scVI=30,
        prob_based_stopping = False,
        default_input_data='normlog',
        use_feature_selection=True,
        n_top_genes_per_class=300,
        hv_genes=-1,
        sampling_method=None,
        sampling_strategy='auto',
        batch_key='batch', 
        scVI_model_prefix=None):

        self.save_path = save_path
        self.n_dimensions_scVI = n_dimensions_scVI
        self.prob_based_stopping = prob_based_stopping
        self.default_input_data = default_input_data
        self.use_feature_selection = use_feature_selection
        self.n_top_genes_per_class = n_top_genes_per_class
        self.scVI_model_prefix = scVI_model_prefix
        self.adata = None
        self.dict_of_cell_relations = None
        self.obs_names = None
        self.hv_genes = hv_genes
        if type(sampling_method) != type(None):
            self.init_resampling(sampling_method, sampling_strategy)

        if type(adata) != type(None):
            self.load_adata(adata, batch_key)

        if type(dict_of_cell_relations) != type(None) and type(obs_names) != type(None):
            self.set_cell_relations(dict_of_cell_relations, obs_names)       

    def get_training_data(
        self, 
        node,
        barcodes,
        obs_name_children,
        scVI_key=None):
        """Gets untransformed input and target data for the training of a local classifier, 
        z-transforms the input data (scVI dimensions in the case of NeuralNetwork), calls upon
        self to encode the cell type labels into onehot and integer format using
        the label encoder at the local node.

        Parameters
        ----------
        node
            Specifies which local classifier is currently being trained. This information is
            used to access the relevant label encoder in self.transform_y.
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
        data = self.graph.nodes[node]['local_classifier'].data_type
        return_adata = self.graph.nodes[node]['local_classifier'].input_as_adata
        x, y = self.get_x_y_untransformed(
            barcodes=barcodes, 
            obs_name_children=obs_name_children, 
            data=data, 
            var_names=var_names, 
            scVI_key=scVI_key, 
            return_adata=return_adata)

        if return_adata == False:
            if self.get_preferred_classifier(node) != LogRegWrapper:
                x = z_transform_properties(x)

            else:
                self.fit_chi2_feature_selecter(node, x, y)
                x = self.graph.nodes[node]['chi2_feature_selecter'].transform(x)

            y_int, y_onehot = self.transform_y(node, y)

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
            barcodes = self.adata.obs_names

        obs_name_children = self.get_children_obs_key(node)
        var_names = self.get_selected_var_names(node, barcodes, obs_name_children)
        self.ensure_existence_label_encoder(node)
        self.set_chi2_feature_selecter(node)
        type_classifier = self.get_preferred_classifier(node)
        if type(type_classifier) == type(None):
            type_classifier = NeuralNetwork

        scVI_key = None
        if self.graph.nodes[node]['local_classifier'].data_type == 'scVI':
            scVI_node = self.node_to_scVI[node]
            scVI_key = self.get_scVI_key(
                node=scVI_node, 
                n_dimensions=self.n_dimensions_scVI,
                barcodes=barcodes)

        x, y, y_int, y_onehot = self.get_training_data(node, barcodes, obs_name_children, scVI_key=scVI_key)
        if self.graph.nodes[node]['local_classifier'].data_type == 'scVI':
            input_n_classifier = self.n_dimensions_scVI

        elif type(var_names) == type(None):
            input_n_classifier = len(self.adata.var_names)

        else:
            input_n_classifier = len(var_names)

        self.ensure_existence_classifier(
            node, 
            input_n_classifier,
            classifier=type_classifier)
        self.graph.nodes[node]['local_classifier'].train(x=x, y_onehot=y_onehot, y=y, y_int=y_int)
        train_acc, train_con_mat = self.graph.nodes[node]['local_classifier'].validate(x=x, y_int=y_int, y=y)
        self.graph.nodes[node]['last_train_acc'] = train_acc
        self.graph.nodes[node]['last_train_con_mat'] = train_con_mat

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

        obs_name_parent = self.get_parent_obs_key(current_node)
        true_barcodes = self.get_true_barcodes(
            obs_name_parent, 
            current_node, 
            true_from=train_barcodes)
        self.train_single_node(current_node, true_barcodes)
        for child_node in self.get_child_nodes(current_node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            self.train_all_child_nodes(child_node, train_barcodes=train_barcodes)

    def predict_single_node(
        self,
        node,
        barcodes=None):
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

        print(f'Predicting cells at {node}.')
        if type(barcodes) == type(None):
            barcodes = self.adata.obs_names

        print(f'Making prediction for {len(barcodes)} cells based on node assignment and'\
            ' designation as prediction data.')
        if not self.is_trained_at(node):
            raise Exception(f'Must train local classifier for {node} before trying to predict cell'\
                ' types')

        scVI_key = None
        if self.graph.nodes[node]['local_classifier'].data_type == 'scVI':
            scVI_node = self.node_to_scVI[node]
            scVI_key = self.get_scVI_key(
                node=scVI_node, 
                n_dimensions=self.n_dimensions_scVI,
                barcodes=barcodes)

        var_names = self.get_selected_var_names(node, barcodes)
        data = self.graph.nodes[node]['local_classifier'].data_type
        return_adata = self.graph.nodes[node]['local_classifier'].input_as_adata
        print(f'Predicting with {len(var_names) if type(var_names) != type(None) else "all available"} genes')
        type_classifier = self.get_preferred_classifier(node)
        x = self.get_x_untransformed(barcodes, data=data, var_names=var_names, scVI_key=scVI_key, return_adata=return_adata)
        if return_adata == False and not type_classifier == LogRegWrapper:
            x = z_transform_properties(x)

        elif type_classifier == LogRegWrapper:
            x = self.graph.nodes[node]['chi2_feature_selecter'].transform(x)

        if not self.prob_based_stopping:
            if type(self.graph.nodes[node]['local_classifier']) in [CellTypistWrapper, LogRegWrapper]:
                y_pred = self.graph.nodes[node]['local_classifier'].predict(x)

            else:
                y_pred_int = self.graph.nodes[node]['local_classifier'].predict(x)
                y_pred = self.graph.nodes[node]['label_encoder'].inverse_transform(y_pred_int)

            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, barcodes, y_pred)

        #%--------------------------------------------------------------------------------------------------------------------------------------------%#       
        #belongs somewhere in the prediction methds, not sure where yet because of test/training problem
        #%--------------------------------------------------------------------------------------------------------------------------------------------%#

        elif self.prob_based_stopping:
            y_pred = self.predict_single_node_proba(node, x)
            #child_obs_key says at which hierarchy level the predictions have to be saved
            child_obs_key = self.get_children_obs_key(node) 
            parent_obs_key = self.get_parent_obs_key(node)
            self.set_prob_based_predictions(node, child_obs_key, parent_obs_key, barcodes, y_pred, fitted_label_encoder=self.graph.nodes[node]['label_encoder'])

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
            current_barcodes = self.adata.obs_names

        if type(test_barcodes) != type(None):
            current_barcodes = [b for b in current_barcodes if b in test_barcodes]

        self.predict_single_node(current_node, barcodes=current_barcodes)
        obs_key = self.get_children_obs_key(current_node)
        for child_node in self.get_child_nodes(current_node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            child_node_barcodes = self.get_predicted_barcodes(
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
            y_obs = self.obs_names[-1]
        
        y = self.adata.obs[y_obs]
        if type(barcodes) == type(None):
            barcodes = self.adata.obs_names

        if type(k) == type(None):
            if isolate_test_network:
                self_copy = deepcopy(self)

            barcodes_train, barcodes_test = train_test_split(barcodes, test_size=test_size, stratify = y)
            self.train_all_child_nodes(starting_node, barcodes_train)
            self.predict_all_child_nodes(starting_node, test_barcodes=barcodes_test)
            self.get_total_accuracy(y_obs, test_barcodes=barcodes_test)
            #integrate preliminary hierarchical confusion matrix
            # self.get_hierarchical_accuracy(test_barcodes=barcodes_test, level_obs_keys=self.obs_names, all_labels = self.all_nodes, overview_obs_key = 'Level_2' )
            if isolate_test_network:
                self = deepcopy(self_copy)

        else:
            skf = StratifiedKFold(n_splits=k)
            con_mats = []
            accs = []
            for barcodes_train_idx, barcodes_test_idx in skf.split(barcodes, y):
                if isolate_test_network:
                    self_copy = deepcopy(self)

                barcodes_train = barcodes[barcodes_train_idx]
                barcodes_test = barcodes[barcodes_test_idx]
                self.train_all_child_nodes(starting_node, barcodes_train)
                self.predict_all_child_nodes(starting_node, test_barcodes=barcodes_test)
                acc, con_mat, possible_labels = self.get_total_accuracy(y_obs, test_barcodes=barcodes_test)
                # acc, con_mat, possible_labels, con_mat_overview, possible_labels_overview = self.get_hierarchical_accuracy(test_barcodes=barcodes_test, level_obs_keys=self.obs_names, all_labels=self.all_nodes, overview_obs_key = 'Level_2')
                con_mats.append(con_mat)
                # con_mats.append(con_mat_overview)
                accs.append(acc)
                if isolate_test_network:
                    self = deepcopy(self_copy)

            averaged_con_mat = np.sum(con_mats, axis=0) / np.sum(np.sum(con_mats, axis=0), axis=1)
            test_score_mean = ufloat(np.mean(accs), np.std(accs))
            print('Average con mat')
            disp = ConfusionMatrixDisplay(confusion_matrix=averaged_con_mat, display_labels=possible_labels)
            disp.plot(xticks_rotation='vertical')
            print(f'Test accuracy was {test_score_mean}')

    def set_classifier_type(self, node, preferred_classifier):
        if type(node) == list:
            for n in node:
                self.set_preferred_classifier(n, preferred_classifier)

        else:
            self.set_preferred_classifier(node, preferred_classifier)