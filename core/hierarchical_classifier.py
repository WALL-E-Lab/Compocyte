import networkx as nx
import pandas as pd
import numpy as np
import random
import tensorflow as tf 
import tensorflow.keras as keras
import scvi
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from uncertainties import ufloat
from classiFire.core.tools import list_subgraph_nodes, \
z_transform_properties, set_node_to_scVI
from classiFire.core.node_memory import NodeMemory
from classiFire.core.neural_network import NeuralNetwork
from classiFire.core.sequencing_data_container import SequencingDataContainer

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

        # Choose the most cell-type specific scVI dimensions available.
        scVI_node = self.hierarchy_container.node_to_scVI[node]
        scVI_key = self.data_container.get_scVI_key(
            node=scVI_node, 
            n_dimensions=n_dimensions_scVI, 
            node=node, 
            barcodes=barcodes)
        self.hierarchy_container.ensure_existence_classifier(
            node, 
            n_dimensions_scVI,
            classifier=NeuralNetwork)
        self.hierarchy_container.ensure_existence_label_encoder(node)
        if barcodes == None:
            barcodes = self.data_container.adata.obs_names

        obs_name_children = self.hierarchy_container.get_children_obs_key(node)
        x, y_int, y_onehot = self.get_training_data(node, barcodes, scVI_key, obs_name_children)
        self.hierarchy_container.train_single_node(node, x, y_int, y_onehot)

    def predict_all_child_nodes(self,
        current_node=None,
        barcodes=None,
        is_test=False,
        parent_node=None):
        """Add explanation
        """

        if type(barcodes) == type(None):
            barcodes = np.array(self.adata.obs.index)

        # SHOULD BE SET DEPENDING ON CLASSIFIER
        # SHOULD BE A PROPERTY OF THE NODE !!!!!!
        n_dimensions_scVI = 10
        scVI_key = self.get_scVI_key(n_dimensions=n_dimensions_scVI, node=current_node, barcodes=barcodes)
        adata_subset = self.adata[barcodes, :].copy()
        x = adata_subset.obsm[scVI_key]
        x = z_transform_properties(x)
        print(f'Running {current_node}.')
        print(f'Subsetting to {len(adata_subset)} cells based on node assignment and '\
            'designation as prediction data.')

        pred_vec = self.graph.nodes[current_node]['memory'].local_classifier.predict(x)
        try:
            suffix_obs = self.node_to_obs[parent_node]

        except:
            suffix_obs = current_node

        obs_name_pred = f'pred_int_{suffix_obs}'
        self.adata.obs.loc[barcodes, obs_name_pred] = pred_vec
        obs_name_pred = f'pred_{suffix_obs}'
        self.adata.obs.loc[barcodes, obs_name_pred] = self.graph.nodes[current_node]['memory']\
            .label_encoder.inverse_transform(pred_vec)

        for node in self.graph.adj[current_node].keys():
            if len(list(self.graph.adj[node].keys())) == 0:
                continue

            adata_subset_child_node = adata_subset[
                adata_subset.obs[self.node_to_obs[current_node]] == node
            ]
            barcodes_child_node = list(adata_subset_child_node.obs.index)
            self.predict_all_child_nodes(node, barcodes_child_node, is_test=is_test, parent_node=current_node)

        # Continue

    def train_all_child_nodes(self, 
        current_node=None, 
        parent_node=None, 
        test_size=0.2, 
        test_division=True,
        barcodes_train=None,
        barcodes_test=None):
        """Runs all nodes starting at current_node (if supplied), trains local classifiers
        using only those cells that have been annotated as truly belong to that node, e.g.
        the T node classifier is only trained using cells actually classified as T at the relevant
        level.
        """

        if current_node == None:
            current_node = self.dict_of_cell_relations.keys()[0]

        print(f'Running {current_node}.')
        true_node_subset = self.adata.obs
        if parent_node != None:
            true_node_subset = true_node_subset[
                true_node_subset[self.node_to_obs[parent_node]] == current_node
            ]

        if test_division:
            if type(barcodes_train) == type(None) and type(barcodes_test) == type(None):
                barcodes_train, barcodes_test = train_test_split(
                    np.array(self.adata.obs.index), 
                    test_size=test_size, 
                    random_state=42)

            true_node_subset = true_node_subset[
                true_node_subset.index.isin(barcodes_train)
            ]

        print(f'Subsetting to {len(true_node_subset)} cells based on node assignment and '\
            'designation as training data.')
        current_barcodes = list(true_node_subset.index)
        self.run_single_node(current_node, current_barcodes)
        for node in self.graph.adj[current_node].keys():
            if len(list(self.graph.adj[node].keys())) == 0:
                continue

            self.train_all_child_nodes(
                node, 
                parent_node=current_node,
                test_size=test_size,
                test_division=test_division,
                barcodes_train=barcodes_train,
                barcodes_test=barcodes_test)

    def train_local_classifier_kfold_CV(self, node, k=10, sampling_class = SMOTE, sampling_strategy = 'auto', **kwargs):
        """Train and validate local classifier by using stratified k-fold crossvalidation, oversamples 
            training data via SMOTE per default, currently equalizes representation of classes"""

        skf = StratifiedKFold(n_splits=k)
        X = self.graph.nodes[node]['memory']._get_raw_x_input_data()
        y = self.graph.nodes[node]['memory']._get_raw_y_input_data()
        output_len = self.graph.nodes[node]['memory']._get_output_len_of_node() 

        train_scores = []
        test_scores = []
        train_conmat=[]
        test_conmat=[]

        for train_index, test_index in skf.split(X, y):
            X_train, y_train_int, y_train_onehot = self.graph.nodes[node]['memory']._get_indexed_data(train_index)

            if sampling_class != None:
                sampler = sampling_class(sampling_strategy=sampling_strategy)
                X_train, y_train_int = sampler.fit_transform(X_train, y_train_int)
                #cave: watch out if keras really keeps the same encoding system
                y_train_onehot = keras.utils.to_categorical(y_train_int)

            X_test, y_test_int, y_test_onehot = self.graph.nodes[node]['memory']._get_indexed_data(test_index)
            self.init_local_classifier(
                node, 
                NeuralNetwork, 
                X_train, 
                y_train_onehot, 
                output_len, 
                **kwargs)
            self.graph.nodes[node]['memory'].local_classifier.train()

            train_acc, train_con_mat = self.graph.nodes[node]['memory'].local_classifier.validate(X_train, y_train_int) 
            train_scores.append(train_acc) 
            train_conmat.append(train_con_mat)

            test_acc, test_con_mat = self.graph.nodes[node]['memory'].local_classifier.validate(X_test, y_test_int)
            test_scores.append(test_acc) 
            test_conmat.append(test_con_mat)

            self.graph.nodes[node]['memory']._set_trainings_accuracy(
                ufloat(np.mean(train_scores), np.std(train_scores)))
            self.graph.nodes[node]['memory']._set_test_accuracy(
                ufloat(np.mean(test_scores), np.std(test_scores)))

            self.graph.nodes[node]['memory']._set_training_conmat(train_conmat)
            self.graph.nodes[node]['memory']._set_test_conmat(test_conmat)


    def local_classifier_prediction(self, node, x_data):
        """Predict cell types of each cell in dataset
        ______________________________________
        Params:
        --------------------------------------
        node: node, in which local classifier shall be run to predict cell types
        """

        pred_vec = self.graph.nodes[node]['memory'].local_classifier.predict(x_data)
        self.graph.nodes[node]['memory']._set_prediction_vector(pred_vec)
        self.subset_pred_vec(node) 

    def master(self, root):#, x_data, y_data):
        """run HC automated with Neural Network classifiers
        """

        # init and run root node
        y_onehot = self.graph.nodes[root]['memory']._get_y_input_onehot()
        # get_raw_x does not get counts, but the x_input_data to NodeMemory, so
        # scVI data usually
        x_data = self.graph.nodes[root]['memory']._get_raw_x_input_data()
        output_len = self.graph.nodes[root]['memory']._get_output_len_of_node()

        self.init_local_classifier(root, NeuralNetwork, x_data, y_onehot, output_len)
        self.train_local_classifier_kfold_CV(root) 
        x_for_predict = self.graph.nodes[root]['memory']._get_raw_x_input_data()
        self.local_classifier_prediction(root, x_for_predict)

        # run following nodes 
        for node in self.graph.adj[root].keys():
            # anscheinend wird das Objekt an terminalen knoten doch initialisiert, 
            # auch wenn warnung ausgegeben wird, nur processing funktioniert dann 
            # wegen group labels nicht
            if self.graph.nodes[node]['memory'].bool_processed:
                self.master(node)

            else:
                pass

    def grid_search_local_classifier(
        self,
        adata,
        node,
        obs_name,
        options_scVI = [10, 20, 50],
        options_n_neurons = [64, 128, 256],
        options_n_layers = [2, 5, 10],
        options_dropout = [0.1, 0.2, 0.4, 0.6],
        options_learning_rate = [0.002, 0.001, 0.005],
        options_momentum = [0.5, 0.9],
        options_batch_size = [20, 40, 60],
        options_batch_norm = [True, False],
        options_l2_reg = [True, False],
        options_leakiness_ReLU = [0.1, 0.2],
        random_tries = 10,
        n_sample=50000
    ):
        performance_df = pd.DataFrame(columns=[
            'train_accuracy',
            'test_accuracy',
            'scVI_dimensions',
            'layer_structure',
            'dropout',
            'learning_rate',
            'momentum',
            'batch_size',
            'batch_norm',
            'l2_reg',
            'leakiness_ReLU'])
        for _ in range(random_tries):
            print(len(performance_df))
            scVI_dim = random.sample(options_scVI, 1)[0]
            if n_sample > len(list(adata.obs_names)):
                adata_sample = adata
            else:
                adata_sample = adata[adata.obs.sample(n=n_sample, replace=False, random_state=1, axis=0).index,:]
            self.init_node_memory_object(
                node, 
                adata_sample.obsm[f'X_scVI_{scVI_dim}'], 
                adata_sample.obs[obs_name])
            layer_structure = []
            for i in range(random.sample(options_n_layers, 1)[0]):
                layer_structure.append(
                    random.sample(options_n_neurons, 1)[0]
                )

            learning_rate = random.sample(options_learning_rate, 1)[0]
            momentum = random.sample(options_momentum, 1)[0]
            dropout = random.sample(options_dropout, 1)[0]
            batch_size = random.sample(options_batch_size, 1)[0]
            batch_norm = random.sample(options_batch_norm, 1)[0]
            l2_reg = random.sample(options_l2_reg, 1)[0]
            leakiness_ReLU = random.sample(options_leakiness_ReLU, 1)[0]
            self.train_local_classifier_kfold_CV(
                node,
                k=4,
                list_of_hidden_layer_nodes=layer_structure,
                activation_function='relu',
                learning_rate=learning_rate,
                momentum=momentum,
                dropout=dropout,
                batch_size=batch_size,
                batch_norm=batch_norm,
                l2_reg=l2_reg,
                leakiness_ReLU=leakiness_ReLU,
            )
            parameters = {
                'train_accuracy': self.graph.nodes[node]['memory']._get_trainings_accuracy(),
                'test_accuracy': self.graph.nodes[node]['memory']._get_test_accuracy(),
                'scVI_dimensions': scVI_dim,
                'layer_structure': str(layer_structure),
                'dropout': dropout,
                'learning_rate': learning_rate,
                'momentum': momentum,
                'batch_size': batch_size,
                'batch_norm': batch_norm,
                'l2_reg': l2_reg, 
                'leakiness_ReLU': leakiness_ReLU
            }
            index = len(performance_df)
            for key, value in parameters.items():
                performance_df.loc[index, key] = value

        print(performance_df)
        return performance_df

    def group_labels_of_subgraph_to_parent_label(self, super_node):
        """Maps y_input_data labels to parent label (eg map (CD4 T, CD8 T, NK) -> (TNK) for all neighbors of given super_node (eg TNK, B, Others) of Graph g
        _________________________________________________________
        Params:
        ---------------------------------------------------------
        super_node: node at which local classifier has to be run

        _________________________________________________________
        Returns:
        --------------------------------------------------------- 
        vector with mapped labels

        deletes incorrectly labeled cells in Node_Memory object
        """ 

        mapping_dict = {
            node : [child_node for child_node in list_subgraph_nodes(self.graph, node)] \
            for node in self.graph.adj[super_node].keys()}

        # check if some neighbor nodes have already achieved highest annotation depth (terminal nodes) and map those to itself
        for key in mapping_dict.keys():
            if mapping_dict[key] == []:
                mapping_dict[key] = key

        # write dict such that key and value are interchanged, then map
        # https://stackoverflow.com/questions/32262982/pandas-combining-multiple-categories-into-one  
        mapper = {k:v for v,k in pd.Series(mapping_dict).explode().iteritems()} 

        # check if some cells were mislead and throw them out 
        # DOES NOT WORK FOR PREDICTION OF UNLABELED CELLS OF COURSE!
        # OPTION TRAINING/PREDICTION MODE NEEDED!
        idx_of_incorrectly_labelled_cells = []
        for cell_idx, cell_label in enumerate(
            self.graph.nodes[super_node]['memory'].y_input_data
        ):
            if cell_label not in mapper.keys():
                idx_of_incorrectly_labelled_cells.append(cell_idx)

        self.graph.nodes[super_node]['memory']._delete_incorrect_labeled_cells(
            idx_of_incorrectly_labelled_cells)
        y_input_grouped_labels = [
            mapper.get(k) for k in self.graph.nodes[super_node]['memory'].y_input_data]

        return y_input_grouped_labels

    def subset_pred_vec(self, node):
        """subsets raw x_input_data and y_input_data of node into data sets for \
        daughter nodes, based on the prediciton of this nodes classifier
        ________________________________________________
        Params:
        ------------------------------------------------
        node: parent_node from where input data shall be distributed to \
        daughter nodes 

        ________________________________________________
        Returns:
        ------------------------------------------------
        None, initializes following daughter Nodes with subsetted raw x_input \
        and finest y_input of current node
        """

        # List next neighbors of node in order to find the next labels to be predicted
        next_labels = [label for label in self.graph[node].keys()]
        current_pred_vec = self.graph.nodes[node]['memory']._get_prediction_vector()
        current_pred_vec = self.graph.nodes[node]['memory'].label_encoder.inverse_transform(current_pred_vec)

        for next_label in next_labels:
            temp_idx_vec = np.where(current_pred_vec == next_label) 

            subs_x_input = np.array(
                self.graph.nodes[node]['memory']._get_raw_x_input_data())[temp_idx_vec]
            subs_y_input = np.array(
                self.graph.nodes[node]['memory']._get_raw_y_input_data())[temp_idx_vec]

            # De können jetzt verwendet werden um direkt den nächsten Knoten durch zu initialisieren oder aber return diese arrays und 
            # ach initialisierungsaufruf von woanders 

            #initialize the next Node_Memory objects
            try:
                self.init_node_memory_object(next_label, subs_x_input, subs_y_input)

            except:
                print(f'Warning! {next_label} Node could not be initialized!')