from typing import Union
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import robust_scale
from Compocyte.core.base.data_base import DataBase
from Compocyte.core.base.hierarchy_base import HierarchyBase
from Compocyte.core.base.export_import_base import ExportImportBase
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.fit_methods import fit, predict
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.models.dense_torch import DenseTorch
from time import time
from scipy import sparse
import numpy as np
import os
import pickle
import scanpy as sc
import multiprocessing as mp
from Compocyte.core.models.trees import BoostedTrees
from Compocyte.core.tools import infer_dict, z_transform_properties


class HierarchicalClassifier(
        DataBase,
        HierarchyBase,
        ExportImportBase):
    def __init__(
            self,
            save_path,
            adata=None,
            root_node=None,
            dict_of_cell_relations=None, 
            obs_names=None,
            default_input_data='normlog',
            num_threads=1,
            ignore_counts=False,
            temp_paths=None,
            graph=None,
            intermittent_saving=False,
            ):
        """Initialize a HierarchicalClassifier.

        Args:
            save_path (str): Directory where classifier state is saved and loaded from.
            adata (sc.AnnData, optional): AnnData object containing single-cell data.
            root_node (str, optional): Name of the root node in the cell-type hierarchy.
            dict_of_cell_relations (dict, optional): Nested dictionary defining the
                cell-type hierarchy, e.g.
                ``{'root': {'T cell': {'CD4+': {}, 'CD8+': {}}}}``.
            obs_names (list of str, optional): List of ``adata.obs`` column names
                indexed by hierarchy depth (index 0 = root level, index 1 = first child,
                …).
            default_input_data (str): Input data representation to use. One of
                ``'normlog'`` (log-normalized counts in ``adata.X``) or ``'counts'``
                (raw counts). Defaults to ``'normlog'``.
            num_threads (int): Number of CPU threads to use per training process.
                Defaults to ``1``.
            ignore_counts (bool): If ``True``, the presence of raw count data is not 
                confirmed. Must be used, for example, if .X contains 
                dimensionality-reduced or manually normalized data. Defaults to ``False``.
            temp_paths (str or list of str, optional): Temporary directory paths used
                when importing or exporting classifier model files.
            graph (networkx.DiGraph, optional): Pre-built hierarchy graph. When
                provided, ``dict_of_cell_relations`` is inferred from it via
                ``infer_dict``.
            intermittent_saving (bool): If ``True``, the classifier state is saved to
                disk after each node is trained. Defaults to ``False``.

        Example:
            >>> hc = HierarchicalClassifier(
            ...     save_path='/data/classifier',
            ...     adata=adata,
            ...     root_node='root',
            ...     dict_of_cell_relations={'root': {'T cell': {}, 'B cell': {}}},
            ...     obs_names=['cell_type_l1', 'cell_type_l2'],
            ... )
        """

        if graph is None and dict_of_cell_relations is None:
            print('Neither graph nor dict_of_cell_relations defined upon initialization.')
            print('Please run .load() to load an existing classifier.')

        elif obs_names is None or root_node is None:
            print('obs_names and root_node must be defined upon initialization if a new classifier is initialized.')

        self.save_path = save_path
        self.default_input_data = default_input_data
        self.num_threads = num_threads
        self.adata = None
        self.var_names = None
        self.dict_of_cell_relations = None
        self.root_node = None
        self.obs_names = None
        self.ignore_counts = ignore_counts
        self.intermittent_saving = intermittent_saving

        if dict_of_cell_relations is None and graph is not None:
            dict_of_cell_relations = infer_dict(graph)

        if type(adata) != type(None):
            self.load_adata(adata)

        if root_node is not None and dict_of_cell_relations is not None and obs_names is not None:
            self.set_cell_relations(root_node, dict_of_cell_relations, obs_names, temp_paths)

    def save(self, save_adata=False):
        """Save the classifier state to disk.

        Serializes instance attributes, per-node graph content, and local classifiers
        to timestamped subdirectories under ``self.save_path``. Each call creates a
        new timestamped snapshot; loading always restores the most recent snapshot.

        Args:
            save_adata (bool): If ``True``, also writes ``self.adata`` to an ``.h5ad``
                file under ``<save_path>/data/``. Defaults to ``False``.

        Example:
            >>> hc.save()
        """
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
                if self.adata is None or not save_adata:
                    continue
                
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

                    local_classifier = self.graph.nodes[node]['local_classifier']
                    if isinstance(local_classifier, DenseTorch) or isinstance(local_classifier, LogisticRegression) or isinstance(local_classifier, DummyClassifier) or isinstance(local_classifier, BoostedTrees):
                        self.graph.nodes[node]['local_classifier']._save(model_path)
                        
                    continue

                with open(os.path.join(node_content_path, f'{key}.pickle'), 'wb') as f:
                    pickle.dump(self.graph.nodes[node][key], f)

    def load(self, load_path=None, load_adata=False):
        """Load the most recent classifier snapshot from disk.

        Restores instance attributes from the latest timestamped pickle under
        ``<load_path>/hierarchical_classifiers/``, optionally reloads ``adata``, and
        deserializes per-node graph content and local classifiers. Reconstructs
        ``self.graph`` if it does not already exist.

        Args:
            load_path (str, optional): Root directory to load from. Defaults to
                ``self.save_path``.
            load_adata (bool): If ``True``, loads the most recent ``.h5ad`` file from
                ``<load_path>/data/`` into ``self.adata``. Defaults to ``False``.

        Example:
            >>> hc = HierarchicalClassifier(save_path='/data/classifier')
            >>> hc.load()
            >>> hc.load(load_path='/backup/classifier', load_adata=True)
        """
        if load_path is None:
            load_path = self.save_path
            
        data_path = os.path.join(
            load_path, 
            'data'
        )
        hc_path = os.path.join(
            load_path, 
            'hierarchical_classifiers'
        )
        if os.path.exists(hc_path):
            timestamps = os.listdir(hc_path)
            last_timestamp = sorted(timestamps)[-1]
            with open(os.path.join(hc_path, last_timestamp, 'hierarchical_classifier_settings.pickle'), 'rb') as f:
                settings_dict = pickle.load(f)
                for key in settings_dict.keys():
                    self.__dict__[key] = settings_dict[key]

        if os.path.exists(data_path) and load_adata:
            timestamps = os.listdir(data_path)
            last_adata = sorted(timestamps)[-1]
            adata = sc.read_h5ad(os.path.join(data_path, last_adata))
            self.load_adata(adata)

        if not hasattr(self, 'graph') or self.graph is None:
            self.make_classifier_graph()

        for node in list(self.graph):
            model_path = os.path.join(
                load_path, 
                'models',
                node
            )
            node_content_path = os.path.join(
                load_path, 
                'node_content',
                node
            )
            if os.path.exists(model_path):
                timestamps = os.listdir(model_path)
                last_timestamp = sorted(timestamps)[-1]
                contents = os.listdir(os.path.join(model_path, last_timestamp))
                if len([c for c in contents if c.startswith('non_param_dict')]) > 0:
                    classifier = DenseTorch._load(os.path.join(model_path, last_timestamp))

                elif 'labels_dec.pickle' in contents and not 'model.cbm' in contents:
                    classifier = LogisticRegression._load(os.path.join(model_path, last_timestamp))

                elif 'labels_dec.pickle' in contents and 'model.cbm' in contents:
                    classifier = BoostedTrees._load(os.path.join(model_path, last_timestamp))

                else:
                    classifier = DummyClassifier._load(os.path.join(model_path, last_timestamp))

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

    def limit_cells(
            self,
            subset: sc.AnnData,
            max_cells: int,
            stratify_by: str) -> sc.AnnData:
        """Randomly subsample cells from a subset, stratified by a given obs column if
        a maximum cell count (max_cells) is exceeded.

        Cells are sampled equally from each stratum defined by ``stratify_by``. If the
        total after stratified sampling falls below ``max_cells``, additional cells are
        drawn at random from the remaining pool to reach the cap.

        Args:
            subset (sc.AnnData): AnnData slice to subsample.
            max_cells (int): Maximum number of cells to retain.
            stratify_by (str): ``adata.obs`` column name used to define strata
                (e.g. ``'dataset'``).

        Returns:
            sc.AnnData: Subsampled AnnData with at most ``max_cells`` observations.
                Returns the original ``subset`` unchanged if
                ``len(subset) <= max_cells``.

        Example:
            >>> limited = hc.limit_cells(
            ...     subset, max_cells=10_000, stratify_by='dataset'
            ... )
        """
        if len(subset) > max_cells:
            rng = np.random.default_rng(42)
            datasets = subset.obs[stratify_by].unique()
            cells_per_dataset = max_cells // len(datasets)
            limited_indices = []

            for dataset in datasets:
                dataset_indices = subset.obs[subset.obs['dataset'] == dataset].index
                if len(dataset_indices) > cells_per_dataset:
                    dataset_indices = rng.choice(dataset_indices, cells_per_dataset, replace=False)
                    
                limited_indices.extend(dataset_indices)

            # If we have fewer cells than max_cells, randomly sample additional cells to reach max_cells
            if len(limited_indices) < max_cells:
                additional_indices = rng.choice(
                    subset.obs.index.difference(limited_indices), 
                    max_cells - len(limited_indices), 
                    replace=False)
                limited_indices.extend(additional_indices)

            subset = subset[limited_indices, :]

        return subset

    def introduce_limit(self, max_cells: int, stratify_by: str):
        """Set a per-training-run cell cap and stratification column.

        The limit is applied during :meth:`select_subset` whenever ``max_cells`` is
        passed to that call. Cells are subsampled via :meth:`limit_cells` using the
        specified ``stratify_by`` column.

        Args:
            max_cells (int): Maximum number of cells per local classifier training run.
            stratify_by (str): ``adata.obs`` column name used to stratify subsampling
                (e.g. ``'dataset'``).

        Example:
            >>> hc.introduce_limit(max_cells=50_000, stratify_by='dataset')
        """

        self.max_cells = max_cells
        self.stratify_by = stratify_by

    def select_subset(
            self,
            node: str,
            features: list=None,
            max_cells: int=None) -> sc.AnnData:
        """Select training cells for a given hierarchy node.

        Returns cells labeled as ``node`` at the node's depth level that also carry a
        non-empty child-level label (i.e. cells with a known subtype). Optionally
        restricts to a specified feature set and applies a stratified cell count cap.

        Args:
            node (str): Name of the hierarchy node to select cells for.
            features (list of str, optional): Gene/feature names to restrict the
                returned subset to. If ``None``, all features are included.
            max_cells (int, optional): Maximum number of cells to return. Only applied
                when :meth:`introduce_limit` has been called first (to set
                ``self.stratify_by``); otherwise ignored.

        Returns:
            sc.AnnData: Subset of ``self.adata`` containing only cells labeled as
                ``node`` with a non-empty child-level annotation.

        Example:
            >>> subset = hc.select_subset('T cell')
            >>> subset = hc.select_subset(
            ...     'T cell', features=['CD3D', 'CD4'], max_cells=10_000
            ... )
        """
        obs = self.obs_names[self.node_to_depth[node]]
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        is_node = self.adata.obs[obs] == node
        has_child_label = self.adata.obs[child_obs] != ''
        subset = self.adata[is_node & has_child_label]
        if features is not None:
            subset = subset[:, features]        

        if max_cells is None:
            max_cells = getattr(self, 'max_cells', None)

        stratify_by = getattr(self, 'stratify_by', None)
        if max_cells is not None and stratify_by is not None:
            subset = self.limit_cells(subset, max_cells, stratify_by)

        return subset
    
    def select_subset_prediction(
            self,
            node: str,
            features: list=None,
            for_trial: bool=False) -> sc.AnnData:
        """Select cells assigned to a given node for inference.

        In normal prediction mode cells are selected by their previously predicted
        label at the node's depth level. When the predicted obs column does not yet
        exist, all cells are returned (root-level fallback). With ``for_trial=True``
        ground-truth labels are used instead of predictions.

        Args:
            node (str): Name of the hierarchy node.
            features (list of str, optional): Gene/feature names to restrict the
                returned subset to. If ``None``, all features are included.
            for_trial (bool): If ``True``, uses ground-truth obs labels instead of
                predicted labels to select cells, to train with all ground-truth relevant 
                cells. Defaults to ``False``.

        Returns:
            sc.AnnData: Subset of ``self.adata`` containing cells associated with
                ``node`` for prediction purposes.

        Example:
            >>> subset = hc.select_subset_prediction('T cell')
            >>> subset = hc.select_subset_prediction('T cell', for_trial=True)
        """

        obs = self.obs_names[self.node_to_depth[node]]
        obs = f'{obs}_pred'
        if obs not in self.adata.obs.columns and not for_trial:
            subset = self.adata

        elif obs not in self.adata.obs.columns and for_trial:
            is_node = self.adata.obs[self.obs_names[self.node_to_depth[node]]] == node
            subset = self.adata[is_node]

        else:
            is_node = self.adata.obs[obs] == node
            subset = self.adata[is_node]

        if features is not None:
            subset = subset[:, features]

        return subset

    def run_feature_selection(
            self,
            node: str,
            overwrite: bool=False,
            n_features: int=-1,
            max_features: int=None,
            min_features: int=30,
            test_factor: float=1.0,
            max_cells=100_000):
        
        """Select the most informative features for classifying children of a node.

        Uses ANOVA F-statistics (``SelectKBest`` with ``f_classif``) on
        robustly-scaled expression values to rank and select features. The target
        feature count is either specified directly via ``n_features`` or inferred from
        sample count using the heuristic of 1 feature per 100 cells, then clamped to
        ``[min_features, max_features]``.

        Args:
            node (str): Name of the hierarchy node whose children are the prediction
                targets.
            overwrite (bool): If ``True``, overwrite any previously stored feature
                selection for this node. Defaults to ``False``.
            n_features (int): Exact number of features to select. Use ``-1`` to infer
                the count from sample size. Defaults to ``-1``.
            max_features (int, optional): Upper bound on the number of features
                selected. Defaults to the total number of features in ``self.adata``.
            min_features (int): Lower bound on the number of features selected.
                Defaults to ``30``.
            test_factor (float): Multiplier applied to the sample-size-inferred feature
                count. Defaults to ``1.0``.
            max_cells (int): Maximum number of cells used for feature selection.
                Defaults to ``100_000``.

        Returns:
            list of str: Names of the selected features (genes).

        Raises:
            Exception: If features have already been selected for ``node`` and
                ``overwrite`` is ``False``.

        Example:
            >>> features = hc.run_feature_selection('T cell', n_features=200)
            >>> features = hc.run_feature_selection(
            ...     'T cell', overwrite=True, min_features=50, max_features=500
            ... )
        """

        has_features = 'selected_var_names' in self.graph.nodes[node].keys()
        if has_features and not overwrite:
            raise Exception(f'Features have already been selected at {node}.')
        
        subset = self.select_subset(node, max_cells=max_cells)
        x = sparse.csr_matrix.toarray(subset.X)
            
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        if len(subset.obs[child_obs].unique()) <= 1:
            return self.adata.var_names.tolist()
        
        # Rule of thumb from Google's rules of ML:
        # At least 100 samples per feature
        if n_features < 0:
            # test_factor should be taken account during hypopt with reduced sample numbers
            n_features = int(len(subset) / 100 * test_factor)

        n_features = max(min_features, n_features)
        if max_features is None:
            max_features = len(self.adata.var_names)

        n_features = min(n_features, max_features)

        x = np.asarray(x)
        x = robust_scale(x, axis=1, with_centering=False, copy=False, unit_variance=True)
        y = np.array(subset.obs[child_obs])
        selecter = SelectKBest(f_classif, k=n_features)
        selecter.fit(x, y)
        features = self.adata.var_names[selecter.get_support()]

        return features.tolist()

    def create_local_classifier(
            self,
            node: str,
            overwrite: bool=False,
            classifier_type: Union[DenseTorch, LogisticRegression, BoostedTrees]=DenseTorch,
            **classifier_kwargs):
        """Instantiate and attach a local classifier to a hierarchy node.

        Creates a classifier of the specified type using the node's previously selected
        features and unique child labels. If the node has only one child label, a
        ``DummyClassifier`` is created regardless of ``classifier_type``.

        Args:
            node (str): Name of the hierarchy node.
            overwrite (bool): If ``True``, replace any existing classifier at this
                node. Defaults to ``False``.
            classifier_type (type or str): Classifier class to instantiate, or one of
                the strings ``'DenseTorch'``, ``'LogisticRegression'``, or
                ``'BoostedTrees'``. Defaults to ``DenseTorch``.
            **classifier_kwargs: Additional keyword arguments forwarded to the
                classifier constructor (e.g. ``hidden_layers``, ``dropout``).

        Raises:
            Exception: If a classifier already exists at ``node`` and ``overwrite``
                is ``False``.
            Exception: If :meth:`run_feature_selection` has not been called for
                ``node`` first.
            Exception: If ``classifier_type`` is an unrecognized string.

        Example:
            >>> hc.create_local_classifier(
            ...     'T cell', classifier_type='DenseTorch',
            ...     hidden_layers=[64, 32], dropout=0.2
            ... )
        """
        has_classifier = 'local_classifier' in self.graph.nodes[node].keys()
        if has_classifier and not overwrite:
            raise Exception(f'A classifier already exists at {node}.')
        
        features = self.graph.nodes[node].get('selected_var_names', None)
        if features is None:
            raise Exception(f'Cannot create classifier at {node} without features.\
                            Please run run_feature_selection first.')
        
        subset = self.select_subset(node)
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        labels = subset.obs[child_obs].unique().tolist()
        n_input = len(features)
        n_output = len(labels)
        if isinstance(classifier_type, str):
            if classifier_type == 'DenseTorch':
                classifier_type = DenseTorch

            elif classifier_type == 'LogisticRegression':
                classifier_type = LogisticRegression

            elif classifier_type == 'BoostedTrees':
                classifier_type = BoostedTrees

            else:
                raise Exception(f'Unknown classifier type: {classifier_type}')
            
        if n_output == 1:
            classifier_type = DummyClassifier

        local_classifier = classifier_type(
            labels,
            n_input=n_input,
            n_output=n_output,
            **classifier_kwargs)
        self.graph.nodes[node]['local_classifier'] = local_classifier


    def train_single_node(
            self,
            node: str,
            standardize_separately: str=None,
            **fit_kwargs):
        """Train the local classifier at a single hierarchy node.

        Runs feature selection and classifier creation if they have not been performed
        yet, then fits the model on cells labeled as ``node``. When
        ``self.tuned_kwargs`` contains an entry for ``node``, those hyperparameters
        override any provided ``fit_kwargs``. Returns ``None`` when the node has fewer
        than 5 labeled cells.

        The method returns a dict of updated node parameters rather than modifying
        ``self.graph`` in-place, which is required for safe use with
        ``multiprocessing.Pool``.

        Args:
            node (str): Name of the hierarchy node to train.
            standardize_separately (str, optional): ``adata.obs`` column name
                (e.g. ``'dataset'``). When provided, cells are grouped by unique
                values of this column and each group is robustly scaled independently
                before training.
            **fit_kwargs: Additional keyword arguments forwarded to
                :func:`~Compocyte.core.models.fit_methods.fit` (e.g. ``epochs``,
                ``batch_size``, ``starting_lr``).

        Returns:
            dict or None: Dictionary of updated node parameters including
                ``'learning_curve'``, or ``None`` if the node has too few cells.

        Raises:
            Exception: If ``num_threads`` is not set on the instance and not provided
                in ``fit_kwargs``.

        Example:
            >>> params = hc.train_single_node('T cell', epochs=50, batch_size=256)
        """
        if not hasattr(self, 'num_threads') and not 'num_threads' in fit_kwargs:
            raise Exception('Please specify the number of threads to use for training.')
        
        elif 'num_threads' in fit_kwargs:
            self.num_threads = fit_kwargs['num_threads']

        has_classifier = 'local_classifier' in self.graph.nodes[node].keys()
        # This weird approach is currently necessary to allow for training with mp.pool
        if hasattr(self, 'tuned_kwargs') and node in self.tuned_kwargs:
            kwargs = self.tuned_kwargs[node]
            features_kwargs = {
                'n_features': kwargs['n_features']
            }
            classifier_kwargs = {
                'hidden_layers': kwargs['hidden_layers'],
                'dropout': kwargs['dropout'],
            }
            fit_kwargs = {
                'epochs': kwargs['epochs'],
                'batch_size': kwargs['batch_size'],
                'starting_lr': kwargs['starting_lr'],
                'max_lr': kwargs['max_lr'],
                'momentum': kwargs['momentum'],
                'beta': kwargs['beta'],
                'gamma': kwargs['gamma'],
                'max_cells': getattr(self, 'max_cells', 1_000_000)
            }
            self.graph.nodes[node]['threshold'] = kwargs['threshold']

        else:
            features_kwargs = {}
            classifier_kwargs = {}

        if not has_classifier:
            subset = self.select_subset(node)
            if len(subset) < 5:
                return
            
            features = self.run_feature_selection(node, **features_kwargs)
            self.graph.nodes[node]['selected_var_names'] = features
            classifier_type = DenseTorch
            hidden_layers = classifier_kwargs.get('hidden_layers', [])
            if -1 in hidden_layers:
                classifier_type = BoostedTrees
                
            # If classifier types other than the standard have been set, use those
            specified_classifier_types =  getattr(self, 'specified_classifier_types', {})
            classifier_type = specified_classifier_types.get(node, classifier_type)
            self.create_local_classifier(node, classifier_type=classifier_type, **classifier_kwargs)
        
        child_obs = self.obs_names[self.node_to_depth[node] + 1]
        features = self.graph.nodes[node]['selected_var_names']
        subset = self.select_subset(node, features=features)
        if len(subset) == 0:
                return
        
        model = self.graph.nodes[node]['local_classifier']
        x = subset.X
        y = subset.obs[child_obs].values
        print(f'Training at {node}.')
        
        if standardize_separately is not None:
            idx = []
            for dataset in subset.obs[standardize_separately].unique():
                idx.append(np.where(subset.obs[standardize_separately] == dataset))

        else:
            idx = None
            
        if not 'max_cells' in fit_kwargs:
            fit_kwargs['max_cells'] = getattr(self, 'max_cells', 1_000_000)

        if not 'num_threads' in fit_kwargs:
            fit_kwargs['num_threads'] = self.num_threads
            
        # Necessary to avoid data loss when using mp.pool
        return {
            **self.graph.nodes[node],
            'learning_curve': fit(model, x, y, standardize_idx=idx, **fit_kwargs)
        }
    
    def set_classifier_type(self, node, classifier_type):
        """Override the default classifier type for one or more hierarchy nodes.

        Stores the mapping in ``self.specified_classifier_types``, which is consulted
        by :meth:`train_single_node` when creating local classifiers. Can be called
        with a single node name or a list of node names.

        Args:
            node (str or list of str): Node name(s) for which to set the classifier
                type.
            classifier_type (type): Classifier class to use at the specified node(s),
                e.g. ``LogisticRegression`` or ``BoostedTrees``.

        Example:
            >>> hc.set_classifier_type('T cell', LogisticRegression)
            >>> hc.set_classifier_type(['B cell', 'NK cell'], BoostedTrees)
        """
        if isinstance(node, list):
            for n in node:
                self.set_classifier_type(n, classifier_type)

        else:
            if not hasattr(self, 'specified_classifier_types'):
                self.specified_classifier_types = {}

            self.specified_classifier_types[node] = classifier_type

    def train_all_child_nodes(
        self,
        parallelize: bool=False,
        processes: int=None) -> None:
        """Train local classifiers for all non-leaf nodes in the hierarchy.

        Iterates over every node that has at least one child and calls
        :meth:`train_single_node`. In sequential mode, saves the classifier state
        after each node when ``self.intermittent_saving`` is ``True``. In parallel
        mode, node parameters are collected after all workers finish and written back
        to the graph.

        Args:
            parallelize (bool): If ``True``, train nodes in parallel using
                ``multiprocessing.Pool``. Defaults to ``False``.
            processes (int, optional): Total number of worker processes. Required when
                ``parallelize=True``. Automatically reduced when
                ``self.num_threads > 1`` to avoid CPU oversubscription.

        Returns:
            None

        Raises:
            Exception: If ``parallelize=True`` and ``processes`` is not specified.

        Example:
            >>> hc.train_all_child_nodes()
            >>> hc.train_all_child_nodes(parallelize=True, processes=8)
        """
        nodes_to_train = []
        for node in self.graph.nodes:
            n_children = len(list(self.graph.successors(node)))
            if n_children >= 1:
                nodes_to_train.append(node)

        if not parallelize:
            for node in nodes_to_train:
                learning_curve = self.train_single_node(node, parallelize=False)['learning_curve']
                self.graph.nodes[node]['learning_curve'] = learning_curve
                if self.intermittent_saving:
                    self.save()
                    
        else: 
            if processes is None:
                raise Exception('Please specify the number of processes to use for parallelization.')
            
            # When setting num_threads > 1 per training process, the number of processes should be limited
            if self.num_threads is not None:
                processes = int(processes / self.num_threads)

            print(f"Using multiprocessing for training with {mp.cpu_count()} available CPU cores.\n")           
            with mp.Pool(processes=processes) as pool: 
                all_trained_node_params = pool.map(self.train_single_node, nodes_to_train)
            
            for node, params in zip(nodes_to_train, all_trained_node_params):
                if params is not None: #this should only happen at nodes that have not been trained
                    for key in params.keys():         
                        if params.get(key) is not None:    
                            self.graph.nodes[node][key] = params.get(key)

    def predict_single_node(
        self,
        node: str,
        threshold: float=-1,
        monte_carlo: int=None) -> np.ndarray:
        """Run inference at a single hierarchy node and write predictions to adata.obs.

        Selects cells currently routed to ``node``, runs the node's local classifier,
        and writes predicted child-level labels to ``<child_obs>_pred`` in
        ``self.adata.obs``. If an ``'overclustering'`` column is present, predictions
        are harmonized per cluster by majority vote.

        When ``monte_carlo`` is set, the method also computes per-sample mean and
        standard deviation of the winning label's activation across MC iterations and
        stores them in ``self.adata.obs['monte_carlo_mean']`` and
        ``self.adata.obs['monte_carlo_std']``.

        Args:
            node (str): Name of the hierarchy node at which to run prediction.
            threshold (float): Minimum confidence required to assign a label. Use
                ``-1`` to disable thresholding and always assign the top-scoring
                label. Defaults to ``-1``.
            monte_carlo (int, optional): Number of Monte Carlo dropout forward passes
                for uncertainty estimation. If ``None``, standard deterministic
                inference is used.

        Returns:
            numpy.ndarray: Array of predicted child-level cell-type labels for cells
                routed to ``node``.

        Example:
            >>> pred = hc.predict_single_node('T cell')
            >>> pred = hc.predict_single_node('T cell', threshold=0.9, monte_carlo=100)
        """

        if 'local_classifier' not in self.graph.nodes[node]:
            return []
        
        features = self.graph.nodes[node]['selected_var_names']
        subset = self.select_subset_prediction(node, features=features)
        if len(subset) == 0:
            return
        
        model = self.graph.nodes[node]['local_classifier']
        x = subset.X
        print(f'Predicting at {node}.')

        pred = predict(model, x, threshold=threshold, monte_carlo=monte_carlo)
        all_logits = None
        if monte_carlo is not None and isinstance(pred, tuple):
            pred, all_logits = pred
            if len(all_logits.shape) < 3:
                all_logits = np.expand_dims(all_logits, axis=1)

        
        if 'overclustering' in subset.obs.columns:
            for cluster_name in subset.obs['overclustering'].unique():
                cluster_indices = subset.obs['overclustering'] == cluster_name
                if np.sum(cluster_indices) > 0:
                    cluster_preds = pred[cluster_indices]
                    if len(cluster_preds) > 0:
                        most_common = max(set(cluster_preds), key=list(cluster_preds).count)
                        pred[cluster_indices] = most_common

        child_obs = self.obs_names[self.node_to_depth[node] + 1] 
        child_obs = f'{child_obs}_pred'
        if child_obs not in self.adata.obs.columns:
            self.adata.obs[child_obs] = ''
            
        self.adata.obs[child_obs] = self.adata.obs[child_obs].astype(str)
        self.adata.obs.loc[
            subset.obs_names,
            child_obs
        ] = pred
        if monte_carlo is not None and all_logits is not None:
            # all_logits: Shape: (iterations, samples, labels)
            # mean_activations_per_sample: Shape: (samples, labels)
            mean_activations_per_sample = np.mean(all_logits, axis=0)
            # idx_max_activations: Shape: (samples)
            idx_max_activations = np.argmax(mean_activations_per_sample, axis=1)
            # idx_tile: Shape: (iterations, samples)
            idx_tile = np.tile(idx_max_activations, (all_logits.shape[0], 1))
            # Expand idx_tile dims to match test dimensions
            # For each iteration and sample, take the activation corresponding to
            # the label with the highest mean activation across iterations for this sample
            # activations_chosen_label_per_iteration: Shape: (iterations, samples, 1)
            activations_chosen_label_per_iteration = np.take_along_axis(
                all_logits, 
                np.expand_dims(idx_tile, axis=2), axis=2)
            # activations_chosen_label_per_sample: Shape: (samples, iterations)
            activations_chosen_label_per_sample = np.squeeze(activations_chosen_label_per_iteration).T
            # when dealing with single samples, activations are squeezed to 1D
            axis = 0 if activations_chosen_label_per_sample.ndim == 1 else 1
            # mean_activation_chosen_label_per_sample: Shape: (samples)
            mean_activation_chosen_label_per_sample = np.mean(activations_chosen_label_per_sample, axis=axis)
            std_activations_chosen_label_per_sample = np.std(activations_chosen_label_per_sample, axis=axis)
            self.adata.obs.loc[
                subset.obs_names,
                'monte_carlo_mean',
            ] = mean_activation_chosen_label_per_sample
            self.adata.obs.loc[
                subset.obs_names,
                'monte_carlo_std',
            ] = std_activations_chosen_label_per_sample

        return pred    

    def predict_all_child_nodes(
        self,
        node: str,
        threshold: float=-1,
        mlnp: bool=False,
        monte_carlo: int=None):
        """Recursively predict cell types from a starting node down the hierarchy.

        Calls :meth:`predict_single_node` at ``node``, then recurses into each child
        that itself has children (i.e. non-leaf children). Per-node thresholds stored
        in ``self.graph.nodes[node]['threshold']`` take precedence over the
        ``threshold`` argument unless ``mlnp=True``.

        Args:
            node (str): Root of the subtree to predict. Typically ``self.root_node``.
            threshold (float): Default confidence threshold applied at each node.
                Use ``-1`` to always assign a label. Defaults to ``-1``.
            mlnp (bool): Mandatory leaf-node prediction. If ``True``, per-node
                thresholds are ignored and prediction is forced to leaf nodes.
                Defaults to ``False``.
            monte_carlo (int, optional): Number of Monte Carlo dropout iterations,
                forwarded to :meth:`predict_single_node`.

        Example:
            >>> hc.predict_all_child_nodes(hc.root_node)
            >>> hc.predict_all_child_nodes(hc.root_node, threshold=0.9)
            >>> hc.predict_all_child_nodes(hc.root_node, monte_carlo=50)
        """
        # For mandatory leaf node prediction use -1
        if not mlnp:
            threshold = self.graph.nodes[node].get('threshold', threshold)

        self.predict_single_node(node, threshold=threshold, monte_carlo=monte_carlo)
        for child_node in self.get_child_nodes(node):
            if len(self.get_child_nodes(child_node)) == 0:
                continue

            self.predict_all_child_nodes(child_node, threshold=threshold, mlnp=mlnp, monte_carlo=monte_carlo)

