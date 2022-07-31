from classiFire.core.tools import z_transform_properties
from classiFire.core.models.neural_network import NeuralNetwork
from classiFire.core.models.logreg import LogRegWrapper
from classiFire.core.models.single_assignment import SingleAssignment
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from uncertainties import ufloat
from copy import deepcopy
from time import time
import tensorflow.keras as keras
import numpy as np

class CPPNBase():
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
        
        data = self.graph.nodes[node]['local_classifier'].data_type
        var_names = self.get_selected_var_names(node, barcodes, data_type=data)
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

    def train_single_node_CPPN(
        self, 
        node, 
        train_barcodes=None):
        """Trains the local classifier stored at node.

        Parameters
        ----------
        node
            Specifies which local classifier is to be trained. node='T' would result in training
            of the classifier further differentiating between T cells.
        barcodes
            Specifies which cells should be used for training.
        """

        if train_barcodes is None:
            train_barcodes = self.adata.obs_names

        type_classifier = self.get_preferred_classifier(node)
        if type_classifier is None:
            type_classifier = NeuralNetwork

        if self.default_input_data in type_classifier.possible_data_types:
            data_type = self.default_input_data

        else:
            data_type = type_classifier.possible_data_types[0]
            print(f'{self.default_input_data} data is not currently compatible with {type_classifier}. Set to {data_type}')

        parent_obs_key = self.get_parent_obs_key(node)
        children_obs_key = self.get_children_obs_key(node)
        potential_cells = self.adata[self.adata.obs[parent_obs_key] == node]
        # To avoid training with cells that should be reserved for testing, make sure to limit
        # to train_barcodes.
        relevant_cells = self.adata[[b for b in potential_cells.obs_names if b in train_barcodes], :]
        # Cells that are unlabeled (nan or '') are not certain to not belong to the cell type in question
        relevant_cells = self.throw_out_nan(relevant_cells, children_obs_key)
        if data_type == 'normlog':
            self.ensure_normlog()

        n_cell_types = len(relevant_cells.obs[children_obs_key].unique())
        if n_cell_types == 0:
            return

        print(f'Training at {node}.')
        selected_var_names = None
        if not 'cell_types_seen' in self.graph.nodes[node]:
            self.graph.nodes[node]['cell_types_seen'] = []

        # Keeping tracks of which cell types the classifier has encountered to update
        # if desired the selected features
        cell_types_seen = self.graph.nodes[node]['cell_types_seen']
        new_cell_types = [label for label in relevant_cells.obs[children_obs_key].unique() if not label in cell_types_seen]
        self.graph.nodes[node]['cell_types_seen'] = cell_types_seen + new_cell_types
        # Feature selection is only relevant for (transformed) gene data, not embeddings
        if self.use_feature_selection and data_type in ['counts', 'normlog']:
            if 'selected_var_names' in self.graph.nodes[node].keys():
                selected_var_names = self.graph.nodes[node]['selected_var_names']
                if len(new_cell_types) > 0 and self.update_feature_selection and n_cell_types > 1:
                    for label in new_cell_types:
                        positive_cells = relevant_cells[relevant_cells.obs[children_obs_key] == label]
                        negative_cells = relevant_cells[relevant_cells.obs[children_obs_key] != label]
                    
                        selected_var_names_node = self.feature_selection(
                            list(positive_cells.obs_names), 
                            list(negative_cells.obs_names), 
                            data_type, 
                            n_features=self.n_top_genes_per_class, 
                            method='chi2')
                        selected_var_names = selected_var_names + [f for f in selected_var_names_node if f not in selected_var_names]

                    self.graph.nodes[node]['selected_var_names'] = selected_var_names

            # Cannot define relevant genes for prediction if there is no cell group to compare to
            elif n_cell_types > 1:
                selected_var_names = []
                for label in relevant_cells.obs[children_obs_key].unique():
                    positive_cells = relevant_cells[relevant_cells.obs[children_obs_key] == label]
                    negative_cells = relevant_cells[relevant_cells.obs[children_obs_key] != label]
                
                    selected_var_names_node = self.feature_selection(
                        list(positive_cells.obs_names), 
                        list(negative_cells.obs_names), 
                        data_type, 
                        n_features=self.n_top_genes_per_class, 
                        method='chi2')
                    selected_var_names = selected_var_names + [f for f in selected_var_names_node if f not in selected_var_names]

                self.graph.nodes[node]['selected_var_names'] = selected_var_names                

        # Initialize with all available genes as input, banking on mask layer for feature selection
        n_input = len(self.adata.var_names)
        self.ensure_existence_classifier(
            node, 
            n_input,
            classifier=type_classifier)
        # As soon as feature selection is available, adjust the mask layer to silence none-selected input nodes
        if not selected_var_names is None:
            idx_selected = np.isin(np.array(self.adata.var_names), np.array(selected_var_names))
            mask = np.zeros(shape=(n_input))
            mask[idx_selected] = 1
            self.graph.nodes[node]['local_classifier'].update_feature_mask(mask)

        if data_type == 'counts':
            x = relevant_cells[:, selected_var_names].X

        elif data_type == 'normlog':
            x = relevant_cells[:, selected_var_names].layers['normlog']

        else:
            raise Exception('Data type not currently supported.')

        if hasattr(x, 'todense'):
            x = x.todense()

        y = np.array(relevant_cells.obs[children_obs_key])
        if hasattr(self, 'sampling_method') and type(self.sampling_method) != type(None):
            res = self.sampling_method(sampling_strategy=self.sampling_strategy)
            x, y = res.fit_resample(x, y)

        if not 'label_encoding' in self.graph.nodes[node]:
            self.graph.nodes[node]['label_encoding'] = {}

        for label in np.unique(y):
            if label in self.graph.nodes[node]['label_encoding']:
                continue

            idx = len(self.graph.nodes[node]['label_encoding'])
            self.graph.nodes[node]['label_encoding'][label] = idx

        y_int = np.array(
                [self.graph.nodes[node]['label_encoding'][label] for label in y]
            ).astype(int)
        output_len = len(list(self.graph.adj[node].keys()))
        y_onehot = keras.utils.to_categorical(y_int, num_classes=output_len)
        x = z_transform_properties(x)
        self.graph.nodes[node]['local_classifier'].train(x=x, y_onehot=y_onehot, y=y, y_int=y_int)
        train_acc, train_con_mat = self.graph.nodes[node]['local_classifier'].validate(x=x, y_int=y_int, y=y)
        self.graph.nodes[node]['last_train_acc'] = train_acc
        self.graph.nodes[node]['last_train_con_mat'] = train_con_mat
        timestamp = str(time()).replace('.', '_')
        if not node in self.trainings.keys():
            self.trainings[node] = {}

        self.trainings[node][timestamp] = {
            'barcodes': list(relevant_cells.obs_names),
            'node': node,
            'data_type': data_type,
            'type_classifier': type(self.graph.nodes[node]['local_classifier']),
            'var_names': selected_var_names,
            'train_acc': train_acc,
            'train_con_mat': train_con_mat,
        }

    def train_all_child_nodes_CPPN(
        self,
        current_node,
        train_barcodes=None,
        initial_call=True):
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

        self.train_single_node_CPPN(current_node, train_barcodes)
        for child_node in self.get_child_nodes(current_node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            self.train_all_child_nodes_CPPN(child_node, train_barcodes=train_barcodes, initial_call=False)

        if initial_call:
            if not 'overall' in self.trainings.keys():
                self.trainings['overall'] = {}

            timestamp = str(time()).replace('.', '_')
            self.trainings['overall'][timestamp] = {
                'train_barcodes': train_barcodes,
                'current_node': current_node
            }

    def predict_single_node_CPPN(
        self,
        node,
        test_barcodes=None,
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

        print(f'Predicting at parent {node}.')
        if test_barcodes is None:
            test_barcodes = list(self.adata.obs_names)

        parent_obs_key = self.get_parent_obs_key(node)
        if not f'{parent_obs_key}_pred' in self.adata.obs.columns and barcodes is None and not node == self.root_node:
            raise Exception('If previous nodes were not predicted, barcodes for prediction need to \
                be given explicitly.')

        elif node == self.root_node and barcodes is None:
            barcodes = test_barcodes

        if not self.is_trained_at(node):
            print(f'Must train local classifier for {node} before trying to predict cell'\
                ' types')
            return

        potential_cells = self.adata[test_barcodes, :]
        if barcodes is not None:
            potential_cells = self.adata[[b for b in barcodes if b in test_barcodes], :]

        if f'{parent_obs_key}_pred' in self.adata.obs.columns:
            relevant_cells = potential_cells[potential_cells.obs[parent_obs_key] == node]

        else:
            relevant_cells = potential_cells

        if len(relevant_cells) == 0:
            return

        data_type = self.graph.nodes[node]['local_classifier'].data_type
        if type(self.graph.nodes[node]['local_classifier']) != NeuralNetwork:
            raise Exception('CPPN classification mode currently only compatible with neural networks.')

        selected_var_names = list(self.adata.var_names)
        # Feature selection is only relevant for (transformed) gene data, not embeddings
        if self.use_feature_selection and data_type in ['counts', 'normlog']:
            selected_var_names = self.graph.nodes[node]['selected_var_names']

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

        if not self.prob_based_stopping:
            y_pred_int = self.graph.nodes[node]['local_classifier'].predict(x)
            label_decoding = {}
            for key in self.graph.nodes[node]['label_encoding'].keys():
                i = self.graph.nodes[node]['label_encoding'][key]
                label_decoding[i] = key

            y_pred = np.array(
                [label_decoding[i] for i in y_pred_int])
            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, list(relevant_cells.obs_names), y_pred)

        #%--------------------------------------------------------------------------------------------------------------------------------------------%#       
        #belongs somewhere in the prediction methds, not sure where yet because of test/training problem
        #%--------------------------------------------------------------------------------------------------------------------------------------------%#

        elif self.prob_based_stopping:
            y_pred = np.array(self.predict_single_node_proba(node, x))
            y_pred_nan_idx = np.where(np.isnan(y_pred))
            y_pred_not_nan_idx = np.where(np.isnan(y_pred) != True)
            y_pred = y_pred.astype(int)
            label_decoding = {}
            for key in self.graph.nodes[node]['label_encoding'].keys():
                i = self.graph.nodes[node]['label_encoding'][key]
                label_decoding[i] = key

            y_pred_not_nan_str = np.array(
                [label_decoding[i] for i in y_pred[y_pred_not_nan_idx]])
            y_pred = y_pred.astype(str)
            y_pred[y_pred_not_nan_idx] = y_pred_not_nan_str
            y_pred[y_pred_nan_idx] = 'stopped'
            #child_obs_key says at which hierarchy level the predictions have to be saved
            obs_key = self.get_children_obs_key(node)
            self.set_predictions(obs_key, list(relevant_cells.obs_names), y_pred)

        if not node in self.predictions.keys():
            self.predictions[node] = {}

        timestamp = str(time()).replace('.', '_')
        self.predictions[node][timestamp] = {
            'barcodes': barcodes,
            'node': node,
            'data_type': self.graph.nodes[node]['local_classifier'].data_type,
            'type_classifier': type(self.graph.nodes[node]['local_classifier']),
            'var_names': selected_var_names,
        }

    def predict_all_child_nodes_CPPN(
        self,
        current_node,
        current_barcodes=None,
        test_barcodes=None,
        initial_call=True):
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

        self.predict_single_node_CPPN(current_node, barcodes=current_barcodes)
        obs_key = self.get_children_obs_key(current_node)
        for child_node in self.get_child_nodes(current_node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            child_node_barcodes = self.get_predicted_barcodes(
                obs_key, 
                child_node,
                predicted_from=test_barcodes)
            self.predict_all_child_nodes_CPPN(child_node, child_node_barcodes, initial_call=False)

        if initial_call:
            if not 'overall' in self.predictions.keys():
                self.predictions['overall'] = {}

            timestamp = str(time()).replace('.', '_')
            self.predictions['overall'][timestamp] = {
                'test_barcodes': test_barcodes,
                'current_barcodes': current_barcodes,
                'current_node': current_node
            }

    def train_child_nodes_with_validation_CPPN(
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
            self.train_all_child_nodes_CPPN(starting_node, barcodes_train)
            self.predict_all_child_nodes_CPPN(starting_node, test_barcodes=barcodes_test)
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
                self.train_all_child_nodes_CPPN(starting_node, barcodes_train)
                self.predict_all_child_nodes_CPPN(starting_node, test_barcodes=barcodes_test)
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

    def update_hierarchy_CPPN(self, dict_of_cell_relations, root_node=None):
        if not root_node is None:
            self.root_node = root_node

        if dict_of_cell_relations == self.dict_of_cell_relations:
            return

        pass