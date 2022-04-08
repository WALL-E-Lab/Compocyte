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
from imblearn.over_sampling import SMOTE
from uncertainties import ufloat
from classiFire.core.tools import list_subgraph_nodes, \
z_transform_properties, set_node_to_scVI
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
        if type(train_barcodes) == type(None):
            pass

        true_barcodes = self.data_container.get_true_barcodes(
            obs_name_parent, 
            current_node, 
            true_from=train_barcodes)
        self.train_single_node(current_node, true_barcodes)
        for child_node in self.hierarchy_container.get_child_nodes(current_node):
            if len(self.hierarchy_container.get_child_nodes(child_node)) == 0:
                continue

            self.train_all_child_nodes(child_node, train_barcodes=train_barcodes)