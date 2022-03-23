import networkx as nx
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from uncertainties import ufloat
from .tools import make_graph_from_edges, list_subgraph_nodes
from .node_memory import NodeMemory
from .neural_network import NeuralNetwork

class HierarchicalClassifier():
    """Class connects Nodes of Local Classifiers, passes results to children \
    classifiers and forms the final hierarchical classifier
    """ 

    def __init__(self, dict_of_cell_relations):
        """Params
        - dict_of_cell_relations: used for initializing network structure of \
        hierarchical classifier
        """

        self.dict_of_cell_relations = dict_of_cell_relations

    def make_classifier_graph(self):
        """Compute Graph from a given dictionary of cell relationships"""

        self.graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, self.graph)

    def init_node_memory_object(self, node, x_input, y_input):
        """Add memory object to Node node; Node_Memory object organizes all relevant local classifier params
        and run preprocessing methods of NodeMemory object"""
        self.graph.add_node(node, memory=NodeMemory(x_input, y_input))
        # relevant prediction labels for node
        self.graph.nodes[node]['memory']._set_y_input_grouped_labels(
            self.group_labels_of_subgraph_to_parent_label(node))
        # processing of data
        self.graph.nodes[node]['memory']._set_processed_input_data()
        # save output len of local classifier/Node 
        output_len = len(self.graph.adj[node].keys()) 
        self.graph.nodes[node]['memory']._set_output_len_of_node(output_len)

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

    def init_local_classifier(self, node, classifier, x_input_data, y_input_onehot, len_of_output, **kwargs):
        """Initializes local classifier
        ___________________________________________________
        Params:
        ---------------------------------------------------
        node: node where local classifier shall be initialized
        x_input_data: transformed x_data which will be used by the classifier \
        without further processing 
        y_input_onehot: onehot encoded y_input (used for training of NN)
        len_of_output: number of prediction categories, ie dimension of output layer
        **kwargs: any arguments taken by classifier conctructor
        """

        lc = classifier(x_input_data, y_input_onehot, len_of_output, **kwargs)
        self.graph.nodes[node]['memory']._setup_local_classifier(lc)


    def train_local_classifier_kfold_CV(self, node, k=10, **kwargs):
        """Train and validate local classifier by using stratified k-fold crossvalidation"""

        skf = StratifiedKFold(n_splits=k)
        X = self.graph.nodes[node]['memory']._get_raw_x_input_data()
        y = self.graph.nodes[node]['memory']._get_raw_y_input_data()
        output_len = len(self.graph.adj[node].keys()) # könnte man jetzt auch aus Node_Memory abrufen

        train_scores = []
        test_scores = []
        train_conmat=[]
        test_conmat=[]

        for train_index, test_index in skf.split(X, y):
            X_train, y_train_int, y_train_onehot = self.graph.nodes[node]['memory']._get_indexed_data(train_index)
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