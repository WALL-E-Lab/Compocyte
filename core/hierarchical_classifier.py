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
from .tools import make_graph_from_edges, list_subgraph_nodes, is_counts, dict_depth, \
hierarchy_names_unique, flatten_dict, z_transform_properties
from .node_memory import NodeMemory
from .neural_network import NeuralNetwork

class HierarchicalClassifier():
    """Class connects Nodes of Local Classifiers, passes results to children \
    classifiers and forms the final hierarchical classifier
    """ 

    def __init__(self, adata, save_path, dict_of_cell_relations, obs_names, batch_key='batch'):
        """Params
        - adata: AnnData object containing annotations and raw count data
        - dict_of_cell_relations: used for initializing network structure of \
        hierarchical classifier
        - obs_names: list containing the keys for annotations of single cells at the levels
        defined by dict_of_cell_relations. len(obs_names) should be equal to the levels of nesting
        in the dict. I. e. if the parent node for all cells is 'L', and cells are labelled
        as 'L' in obs['Level_1'] then the first entry should be ['Level_1'].
        """

        # Ensure that adata is not a view
        if adata.isview:
            self.adata = adata.copy()

        else:
            self.adata = adata

        self.choose_count_data()
        # Check if the annotations supplied in .obs under obs_names are sufficiently deep
        # to work with the hierarchy provided
        if not dict_depth(dict_of_cell_relations) == len(obs_names):
            raise Exception('obs_names must contain an annotation key for every \
                level of the hierarchy supplied in dict_of_cell_relations.')

        # Check if keys within the hierarchy are unique across all levels as that is a requirement
        # for uniquely identifying graph nodes with networkx
        if not hierarchy_names_unique(dict_of_cell_relations):
            raise Exception('Names given in the hierarchy must be unique.')

        self.save_path = save_path
        self.dict_of_cell_relations = dict_of_cell_relations
        self.obs_names = obs_names
        self.batch_key = batch_key
        self.all_nodes = flatten_dict(self.dict_of_cell_relations)
        self.make_classifier_graph()
        self.set_node_to_obs()
        self.setup_label_encoder()

    def setup_label_encoder(self):
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(np.array(self.all_nodes))

    def set_node_to_obs(self, depth=0, dictionary=None):
        """Create dict assigning to each node in the hierarchy a key in obs under which one
        can find the fitting annotations for the respective sub-nodes. I. e. n cells are labelled
        as T cells in level 2. If one wants to know where the target labels for sub-classification
        of these cells are found, one can call get_obs_from_node('T'), yielding level 3.
        """

        self.node_to_obs = {}
        if dictionary == None:
            dictionary = self.dict_of_cell_relations

        if len(dictionary.keys()) == 0:
            pass

        else:
            for key in dictionary.keys():
                self.set_node_to_obs(depth=depth+1, dictionary=dictionary[key])
                self.node_to_obs[key] = depth + 1

    def get_obs_from_node(self, node):
        """For a given node, returns the obs key under which the annotations to divide into
        sub-groups can be found. I. e. for T_ILC returns the obs key containing annotations like
        T and ILC_ALL.
        """

        return self.node_to_obs[node]

    def choose_count_data(self):
        """Checks adata.X and adata.raw.X for presence of raw count data, setting those up to be used
        in the future.
        """

        if is_counts(self.adata.X):
            self.counts = self.adata.X

        else:
            if hasattr(self.adata, 'raw') and self.adata.raw != None and is_counts(self.adata.raw.X):
                self.counts = self.adata.raw.X

            else:
                raise ValueError('No raw counts found in adata.X or adata.raw.X.')

    def get_scVI_key(n_dimensions=10, node=None, barcodes=None, overwrite=False, **kwargs):
        """Ensure that scVI data is present as requested (specified number of dimensions,
        if applicable for the given node and all of the associated barcodes) If it is not, run scVI.
        Return the obsm key corresponding to the requested scVI data.
        """

        key = f'X_scVI_{n_dimensions}_{"overall" if node == None else node}'
        # Run scVI if it has not been run at the specified number of dimensions or for the specified
        if not key in self.adata.obsm or overwrite:
            self.run_scVI(n_dimensions=n_dimensions, key=key, node=node, barcodes=barcodes, **kwargs)

        elif node != None and barcodes != None:
            # Also run scVI again if entry does not exist for all barcodes supplied
            adata_subset = self.adata[barcodes, :].copy()
            if adata_subset.obs[key].isnull().any():
                self.run_scVI(n_dimensions=n_dimensions, key=key, node=node, barcodes=barcodes, **kwargs)

        return key

    def run_scVI(n_dimensions, key, node=None, barcodes=None, overwrite_scVI=False):
        """Run scVI, currently with parameters taken from the scvi-tools scANVI tutorial. If requested,
        run scVI for a subset of cells only (defined by barcodes, saved by node name).
        """

        scvi.settings.seed = 94705 # For reproducibility
        adata_subset = None
        if node != None and barcodes != None:
            adata_subset = self.adata[barcodes, :].copy()

        # Check if scVI has previously been trained for this node and number of dimensions
        model_path = os.path.join(self.save_path, 'models', 'scvi', key)
        model_exists = os.path.exists(model_path)
        scvi.model.SCVI.setup_anndata(
            adata if adata_subset == None else adata_subset, 
            batch_key=self.batch_key)

        if model_exists and not overwrite_scVI:
            vae = scvi.model.SCVI.load(
                model_path,
                adata if adata_subset == None else adata_subset)

        else:
            arches_params = dict(
                use_layer_norm="both",
                use_batch_norm="none",
                encode_covariates=True,
                dropout_rate=0.2,
                n_layers=2,)
            vae = scvi.model.SCVI(
                adata if adata_subset == None else adata_subset,
                **arches_params)

        vae.train(
            early_stopping=True,
            early_stopping_patience=10)
        vae.save(
            model_path,
            prefix=datetime.now().isoformat(timespec='minutes'),
            overwrite=True)
        self.adata.obsm[key] = vae.get_latent_representation()

    def make_classifier_graph(self):
        """Compute Graph from a given dictionary of cell relationships"""

        self.graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, self.graph)

    def init_node_memory_object(self, node):
        """Add memory object to Node node; Node_Memory object organizes all relevant local classifier params
        and run preprocessing methods of NodeMemory object"""
        
        self.graph.add_node(
            node,
            memory=NodeMemory(self.get_obs_from_node(node)))

    def init_local_classifier(self, node, classifier, input_len, **kwargs):
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

        output_len = len(list(self.graph.adj[node].keys()))
        lc = classifier(input_len, output_len, **kwargs)
        self.graph.nodes[node]['memory']._setup_local_classifier(lc)

    def run_single_node(self, node, barcodes=None, n_dimensions_scVI=10):
        scVI_key = self.get_scVI_key(n_dimensions=n_dimensions_scVI)
        if not node in self.graph.nodes.keys():
            self.init_node_memory_object(node)

        # Check if classifier has been initialized with n_dimensions_scVI
        if not hasattr(self.graph[node]['memory'], 'local_classifier'):
            self.init_local_classifier(node, classifier=neural_network, input_len=n_dimensions_scVI)

        x = None
        y_int = None
        if not barcodes == None:
            adata_subset = self.adata[barcodes, :].copy()
            x = adata_subset.obsm[scVI_key]
            y_int = self.label_encoder.transform(
                np.array(
                    adata_subset.obs[self.node_to_obs(node)]))

        else:
            x = self.adata.obsm[scVI_key]
            y_int = self.label_encoder.transform(
                np.array(
                    self.adata.obs[self.node_to_obs(node)]))

        y_onehot = keras.utils.to_categorical(
            y_int,
            num_classes=len(self.all_nodes))
        x = z_transform_properties(x)
        x_train, x_test, y_onehot_train, y_onehot_test = train_test_split(
            x, y_onehot, test_size=0.2, random_state=42)

        self.graph.nodes[node]['memory'].local_classifier.train()

        train_acc, train_con_mat = self.graph.nodes[node]['memory'].local_classifier.validate(X_train, y_train_int)
        test_acc, test_con_mat = self.graph.nodes[node]['memory'].local_classifier.validate(X_test, y_test_int)
        print(train_acc)
        print(test_acc)
        print(train_con_mat)
        print(test_con_mat)

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