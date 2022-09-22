from classiFire.core.base.data_base import DataBase
from classiFire.core.base.hierarchy_base import HierarchyBase
from classiFire.core.base.CPN_base import CPNBase
from classiFire.core.base.CPPN_base import CPPNBase
from classiFire.core.base.export_import_base import ExportImportBase
from classiFire.core.tools import z_transform_properties
from classiFire.core.models.dense import DenseKeras
from classiFire.core.models.log_reg import LogisticRegression
from classiFire.core.models.dense_torch import DenseTorch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from uncertainties import ufloat
from copy import deepcopy
from imblearn.over_sampling import SMOTE
from time import time
from imblearn.under_sampling import NearMiss, RandomUnderSampler
import tensorflow.keras as keras
import numpy as np
import os
import pickle
import scanpy as sc

class HierarchicalClassifier(DataBase, HierarchyBase, CPNBase, CPPNBase, ExportImportBase):
    """Add explanation
    """

    def __init__(
        self,
        save_path,
        adata = None,
        root_node=None,
        dict_of_cell_relations=None, 
        obs_names=None,
        prob_based_stopping = False,
        threshold=None,
        default_input_data='normlog',
        use_feature_selection=True,
        n_top_genes_per_class=300,
        hv_genes=-1,
        sampling_method=None,
        sampling_strategy='auto',
        batch_key='batch', 
        classification_mode='CPN',
        projected_total_cells=100000,
        sequential_kwargs={}, # hidden_layers learning_rate momentum loss_function dropout discretization l2_reg_input
        train_kwargs={} # batch_size epochs verbose plot
        ):

        self.save_path = save_path
        self.prob_based_stopping = prob_based_stopping
        if threshold is None:
            self.threshold = {'CPPN': 0.9, 'CPN': 0.6}[classification_mode]
            
        else:
            self.threshold = threshold
        self.default_input_data = default_input_data
        self.use_feature_selection = use_feature_selection
        self.n_top_genes_per_class = n_top_genes_per_class
        self.adata = None
        self.dict_of_cell_relations = None
        self.root_node = None
        self.obs_names = None
        self.hv_genes = hv_genes
        self.trainings = {}
        self.predictions = {}
        self.classification_mode = classification_mode
        self.projected_total_cells = projected_total_cells
        self.sequential_kwargs = sequential_kwargs
        self.train_kwargs = train_kwargs
        if type(sampling_method) != type(None):
            self.init_resampling(sampling_method, sampling_strategy)

        if type(adata) != type(None):
            self.load_adata(adata, batch_key)

        if root_node is not None and dict_of_cell_relations is not None and obs_names is not None:
            self.set_cell_relations(root_node, dict_of_cell_relations, obs_names)

    def set_classifier_type(self, node, preferred_classifier):
        if type(node) == list:
            for n in node:
                self.set_preferred_classifier(n, preferred_classifier)

        else:
            self.set_preferred_classifier(node, preferred_classifier)

    def save(self):
        # save all attributes
        # get types, for adata use adatas write function with hash of adata
        # save state of all local classifiers (what does dumping self.graph do?)
        # save state of all nodes in the graph, label encoders, var names ...
        data_path = os.path.join(
            self.save_path, 
            'data'
        )
        timestamp = str(time()).replace('.', '_')
        hc_path = os.path.join(
            self.save_path, 
            'hierarchical_classifiers',
            timestamp
        )
        if not os.path.exists(hc_path):
            os.makedirs(hc_path)

        settings_dict = {}
        for key in self.__dict__.keys():
            if key == 'adata':
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                self.adata.write(os.path.join(data_path, f'{timestamp}.h5ad'))

            elif key == 'graph':
                continue

            else:
                settings_dict[key] = self.__dict__[key]

        with open(os.path.join(hc_path, 'hierarchical_classifier_settings.pickle'), 'wb') as f:
            pickle.dump(settings_dict, f)
        
        for node in list(self.graph):
            node_content_path = os.path.join(
                self.save_path, 
                'node_content',
                node,
                timestamp
            )
            if not os.path.exists(node_content_path):
                os.makedirs(node_content_path)

            for key in self.graph.nodes[node].keys():
                if key == 'local_classifier':
                    model_path = os.path.join(
                        self.save_path, 
                        'models',
                        node,
                        timestamp
                    )
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)

                    if type(self.graph.nodes[node]['local_classifier']) in [DenseKeras, DenseTorch, LogisticRegression]:
                        self.graph.nodes[node]['local_classifier']._save(model_path)
                        
                    continue

                with open(os.path.join(node_content_path, f'{key}.pickle'), 'wb') as f:
                    pickle.dump(self.graph.nodes[node][key], f)

    def load(self):
        data_path = os.path.join(
            self.save_path, 
            'data'
        )
        hc_path = os.path.join(
            self.save_path, 
            'hierarchical_classifiers'
        )
        if os.path.exists(hc_path):
            timestamps = os.listdir(hc_path)
            last_timestamp = sorted(timestamps)[-1]
            with open(os.path.join(hc_path, last_timestamp, 'hierarchical_classifier_settings.pickle'), 'rb') as f:
                settings_dict = pickle.load(f)
                for key in settings_dict.keys():
                    self.__dict__[key] = settings_dict[key]

        if os.path.exists(data_path):
            timestamps = os.listdir(data_path)
            last_adata = sorted(timestamps)[-1]
            adata = sc.read_h5ad(os.path.join(data_path, last_adata))
            self.load_adata(adata, batch_key=self.batch_key)

        if not hasattr(self, 'graph') or self.graph is None:
            self.make_classifier_graph()

        for node in list(self.graph):
            model_path = os.path.join(
                self.save_path, 
                'models',
                node
            )
            node_content_path = os.path.join(
                self.save_path, 
                'node_content',
                node
            )
            if os.path.exists(model_path):
                timestamps = os.listdir(model_path)
                last_timestamp = sorted(timestamps)[-1]
                contents = os.listdir(os.path.join(model_path, last_timestamp))
                if len([c for c in contents if c.endswith('SavedModel')]) > 0:
                    classifier = DenseKeras._load(os.path.join(model_path, last_timestamp))

                elif len([c for c in contents if c.startswith('non_param_dict')]) > 0:
                    classifier = DenseTorch._load(os.path.join(model_path, last_timestamp))

                else:
                    classifier = LogisticRegression._load(os.path.join(model_path, last_timestamp))

                self.graph.nodes[node]['local_classifier'] = classifier

            if os.path.exists(node_content_path):
                timestamps = os.listdir(node_content_path)
                last_timestamp = sorted(timestamps)[-1]
                properties = os.listdir(os.path.join(node_content_path, last_timestamp))
                for p in properties:
                    key = p.replace('.pickle', '')
                    with open(
                        os.path.join(node_content_path, last_timestamp, p), 
                        'rb'
                    ) as f:
                        p = pickle.load(f)
                        self.graph.nodes[node][key] = p

    def train_single_node(self, node, train_barcodes=None, barcodes=None):
        if self.classification_mode == 'CPPN':
            self.train_single_node_CPPN(node, train_barcodes=barcodes)

        elif self.classification_mode == 'CPN':
            self.train_single_node_CPN(node, train_barcodes=train_barcodes)

        else:
            raise ValueError('Classification mode not supported.')

    def train_all_child_nodes(
        self, 
        node,
        train_barcodes=None,
        initial_call=True):
        if self.classification_mode == 'CPPN':
            self.train_all_child_nodes_CPPN(
                node,
                train_barcodes=train_barcodes,
                initial_call=initial_call)

        elif self.classification_mode == 'CPN':
            self.train_all_child_nodes_CPN(
                node,
                train_barcodes=train_barcodes,
                initial_call=initial_call)

        else:
            raise ValueError('Classification mode not supported.')

    def predict_single_node(
        self, 
        node, 
        test_barcodes=None, 
        barcodes=None):
        if self.classification_mode == 'CPPN':
            self.predict_single_node_CPPN(node, barcodes=barcodes)

        elif self.classification_mode == 'CPN':
            self.predict_single_parent_node_CPN(
                node, 
                test_barcodes=test_barcodes, 
                barcodes=barcodes)

        else:
            raise ValueError('Classification mode not supported.')

    def predict_all_child_nodes(
        self, 
        node, 
        test_barcodes=None, 
        initial_call=True, 
        current_barcodes=None):
        if self.classification_mode == 'CPPN':
            self.predict_all_child_nodes_CPPN(
                node,
                current_barcodes=current_barcodes,
                test_barcodes=test_barcodes,
                initial_call=initial_call)

        elif self.classification_mode == 'CPN':
            self.predict_all_child_nodes_CPN(
                node, 
                test_barcodes=test_barcodes, 
                initial_call=initial_call)

        else:
            raise ValueError('Classification mode not supported.')

    def calibrate_single_node(self, node, alpha=0.25, test_barcodes=None, barcodes=None):
        if not self.prob_based_stopping:
            raise Exception('Can only calibrate when using probability based stopping.')

        if not 'local_classifier' in self.graph.nodes[node]:
            print('Cannot calibrate an as yet untrained node.')
            return

        if test_barcodes is None:
            test_barcodes = list(self.adata.obs_names)

        labels = self.get_child_nodes(node)
        if barcodes is None:
            barcodes = list(self.adata[
                (self.adata.obs[self.get_parent_obs_key(node)] == node)
                & (self.adata.obs[self.get_children_obs_key(node)].isin(labels))
                ].obs_names)

        if len(barcodes) == 0:
            return

        if self.classification_mode == 'CPPN':
            used_barcodes, activations = self.predict_single_node_CPPN(node, barcodes=barcodes, get_activations=True)

        elif self.classification_mode == 'CPN':
            used_barcodes, activations = self.predict_single_parent_node_CPN(
                node, 
                test_barcodes=test_barcodes, 
                barcodes=barcodes,
                get_activations=True)
        
        y = np.array(self.adata[used_barcodes, :].obs[self.get_children_obs_key(node)])
        enc = {l: i for i, l in enumerate(labels)}
        y_int = np.array([enc[l] for l in y])
        if len(activations.shape) > 1:
            activations_true = np.take_along_axis(activations.T, y_int[:, np.newaxis], axis=1)[:, 0]

        else:
            activations_true = activations

        conformal_score = 1 - activations_true
        n = y_int.shape[0]
        quantile_n = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(conformal_score, quantile_n)
        threshold = 1 - qhat
        self.graph.nodes[node]['threshold'] = threshold

    def calibrate_all_child_nodes(self, current_node, alpha=0.25):
        """Should only be called after initial training with an initial dataset.
        Can theoretically be called with a holdout dataset from the initial dataset.
        If using probability based stopping, this step is required prior to first prediction.
        It sets the probability activation threshold at which possible labels are added
        to a cells set of plausible predictions. If a set contains no plausible prediction
        or more than one, the cell classification is too uncertain and further classification
        is senseless => probability based stopping of the cell. Alpha determines what error rate
        (i. e. falsely rejecting a cell's true label as part of the set) is deemed acceptable.
        As the calibration is done on a near miss undersampled set of data, 25 % can be assumed
        to lead to much lower false rejection rates overall in majority classes and higher
        false rejection rates in minority classes. This is reasonable and in tune with the
        observation that labels are less certain in general for minority classes.
        Generally inspired by 
        https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf
        """

        if not self.prob_based_stopping:
            raise Exception('Can only calibrate when using probability based stopping.')

        self.calibrate_single_node(current_node, alpha=0.25, test_barcodes=None, barcodes=None)
        for child_node in self.get_child_nodes(current_node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            self.calibrate_all_child_nodes(child_node, alpha=alpha)

    def train_child_nodes_with_validation(
        self, 
        starting_node,
        y_obs=None,
        barcodes=None,
        k=None,
        test_size=0.25,
        isolate_test_network=True):
        if self.classification_mode == 'CPPN':
            self.train_child_nodes_with_validation_CPPN(
                starting_node,
                y_obs=y_obs,
                barcodes=barcodes,
                k=k,
                test_size=test_size,
                isolate_test_network=isolate_test_network)

        elif self.classification_mode == 'CPN':
            raise Exception('Not currently implemented for this classification mode.')

        else:
            raise ValueError('Classification mode not supported.')
