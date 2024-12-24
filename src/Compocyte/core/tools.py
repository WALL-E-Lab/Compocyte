import numpy as np
from scipy.sparse.csr import csr_matrix
import networkx as nx
from sklearn.metrics import ConfusionMatrixDisplay
from copy import deepcopy
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd

def set_node_to_depth(dictionary, depth=0, node_to_depth={}):
    for node in dictionary.keys():
        node_to_depth = set_node_to_depth(dictionary[node], depth=depth+1)
        node_to_depth[node] = depth

    return node_to_depth

def is_counts(matrix, n_rows_to_try=100):
    """Determines whether or not a matrix (such as adata.X, adata.raw.X or an adata layer) contains
    count data by manually checking a subsample of the supplied matrix.
    """

    if not (isinstance(matrix, np.matrix) or isinstance(matrix, csr_matrix) or isinstance(matrix, np.ndarray)):
        raise ValueError(f'Matrix supplied must be of type {csr_matrix} or {np.matrix} or {np.ndarray}.')

    test_data = matrix[:n_rows_to_try]
    if hasattr(test_data, 'todense'):
        test_data = test_data.todense()

    contains_negative_values = np.any(test_data < 0)
    contains_non_whole_numbers = np.any(test_data % 1 != 0)

    return not contains_negative_values and not contains_non_whole_numbers

def dict_depth(dictionary, running_count=0):
    if not type(dictionary) == dict:
        raise TypeError()

    elif len(dictionary.keys()) == 0:
        return running_count

    running_counts_subdicts = []
    for key in dictionary.keys():
        running_counts_subdicts.append(
            dict_depth(
                dictionary[key],
                running_count))

    return max(running_counts_subdicts) + 1

def flatten_dict(dictionary, running_list_of_values=[]):
    if not type(dictionary) == dict:
        raise TypeError()

    elif len(dictionary.keys()) == 0:
        return running_list_of_values

    else:
        for key in dictionary.keys():
            running_list_of_values = running_list_of_values + flatten_dict(dictionary[key]) + [key]

        return running_list_of_values

def hierarchy_names_unique(hierarchy_dict):
    all_nodes = flatten_dict(hierarchy_dict)
    
    return len(all_nodes) == len(set(all_nodes))

def z_transform_properties(data_arr, discretization=False):
    """Calculates a z transformation to center properties across cells in data_arr \
    around mean zero
    """

    mean_vals = np.mean(data_arr, axis=0)
    std_val = np.std(data_arr)
    data_transformed = (data_arr - mean_vals) / std_val
    bin_boundaries = [-0.675, 0, 0.675]
    if discretization:
        data_transformed = np.digitize(data_transformed, bin_boundaries)

    return data_transformed

def make_graph_from_edges(d, g, parent_key=''):
    """Add explanation
    """

    for key in d.keys():
        if parent_key != '':
            g.add_edge(parent_key, key)

        if len(d[key]) == 0:
            pass

        else:
            make_graph_from_edges(d[key], g, parent_key=key)

def get_last_annotation(obs_names, adata, barcodes=None, true_only=False):
    if barcodes is None:
        barcodes = adata.obs_names
        
    obs_names_pred = [f'{x}_pred' for x in obs_names]
    for i, (true_key, pred_key) in enumerate(zip(obs_names, obs_names_pred)):
        if i == 0:
            if true_only:
                obs_df = adata.obs.loc[barcodes, [true_key]]

            else:
                obs_df = adata.obs.loc[barcodes, [true_key, pred_key]]

            obs_df = obs_df[~obs_df[true_key].isin([np.nan, "", "nan"])]
            obs_df.rename(columns={true_key: 'true_last'}, inplace=True)
            if not true_only:
                obs_df = obs_df[~obs_df[pred_key].isin([np.nan, "", "nan"])]
                obs_df.rename(columns={pred_key: 'pred_last'}, inplace=True)

            obs_df = obs_df.astype(str)     

        else:
            if true_only:
                obs_df_level = adata.obs.loc[barcodes, [true_key]]
                obs_df_level.rename(columns={true_key: 'true_last'}, inplace=True)

            else:
                obs_df_level = adata.obs.loc[barcodes, [true_key, pred_key]]
                obs_df_level.rename(columns={true_key: 'true_last', pred_key: 'pred_last'}, inplace=True)

            obs_df_level = obs_df_level.astype(str)
            obs_df_level_true = obs_df_level[~obs_df_level["true_last"].isin([np.nan, "", "nan"])]           
            level_barcodes_true = [x for x in obs_df_level_true.index if x in obs_df.index]
            obs_df.loc[level_barcodes_true, 'true_last'] = obs_df_level_true.loc[level_barcodes_true, 'true_last']
            if not true_only:
                obs_df_level_pred = obs_df_level[~obs_df_level["pred_last"].isin([np.nan, "", "nan"])] 
                level_barcodes_pred = [x for x in obs_df_level_pred.index if x in obs_df.index]
                obs_df.loc[level_barcodes_pred, 'pred_last'] = obs_df_level_pred.loc[level_barcodes_pred, 'pred_last']

    return obs_df

def get_leaf_nodes(hierarchy):
    leaf_nodes = []
    for node in hierarchy.keys():
        if len(hierarchy[node].keys()) != 0:
            leaf_nodes += get_leaf_nodes(hierarchy[node])

        else:
            leaf_nodes += [node]

    return leaf_nodes

def generate_node_level_list(hierarchy, hierarchy_list, level=0):
    for key in hierarchy.keys():
        hierarchy_list.append((key, level))
        generate_node_level_list(hierarchy[key], hierarchy_list, level+1)    

def con_mat_leaf_nodes(hierarchy, adata, graph, obs_names, fig_size=(10, 10), plot=True):
    leaf_nodes = get_leaf_nodes(hierarchy)
    pred_names = [f'{x}_pred' for x in obs_names]
    if type(adata) == sc.AnnData:
        pred_df_true = adata.obs[obs_names].values
        pred_df_pred = adata.obs[pred_names].values

    elif type(adata) == pd.DataFrame:
        pred_df_true = adata[obs_names].values
        pred_df_pred = adata[pred_names].values

    else:
        raise TypeError()

    side_by_side_true_pred = np.dstack([pred_df_true, pred_df_pred])
    hierarchy_list = []
    generate_node_level_list(hierarchy, hierarchy_list)
    true_cells = np.zeros((len(hierarchy_list), 1))
    con_mat = np.zeros((len(hierarchy_list), len(hierarchy_list)))
    for i_true, (key_true, level_true) in enumerate(hierarchy_list[1:]):
        idx_true = np.where(side_by_side_true_pred[:, level_true - 1, 0] == key_true)
        n_true = len(idx_true[0])
        true_cells[i_true + 1, 0] = n_true
        for i_pred, (key_pred, level_pred) in enumerate(hierarchy_list[1:]):
            idx_pred = np.where(side_by_side_true_pred[:, level_pred - 1, 1] == key_pred)
            n_true_pred = len(np.intersect1d(idx_true, idx_pred))
            con_mat[i_true + 1, i_pred + 1] = n_true_pred

    true_cells[np.where(true_cells[:, 0] == 0), 0] = 1
    # Divide by total number of cells truly belonging to each cell type to get proportion of cells
    # of that type assigned to pred label
    con_mat = con_mat / true_cells
    if plot:
        idx_leaf_nodes = np.where(np.isin(np.array(hierarchy_list)[:, 0], leaf_nodes))[0]
        disp = ConfusionMatrixDisplay(con_mat[idx_leaf_nodes, :][:, idx_leaf_nodes], display_labels=np.array(hierarchy_list)[idx_leaf_nodes, 0])
        fig, ax = plt.subplots(figsize=fig_size)
        disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')

    return con_mat

def is_pred_parent_or_child_or_equal(graph, true_label, pred_label, root_node):
    if pred_label in nx.shortest_path(graph, root_node, true_label) or \
    true_label in nx.shortest_path(graph, root_node, pred_label):
        return True

    else:
        return False

def delete_dict_entries(dictionary, del_key='classifier', first_run=True):
    if first_run:
        dictionary = deepcopy(dictionary)

    keys = list(dictionary.keys())
    deleted_key = False
    for key in keys:
        if key == del_key:
            del dictionary[key]
            deleted_key = True

        else:
            deleted_key = deleted_key or delete_dict_entries(dictionary[key], del_key=del_key, first_run=False)

    if first_run:
        return dictionary, deleted_key

    else:
        return deleted_key

def flatten_labels(pred_h_labels, graph, root_node, verbose=False):
    pred_h_labels[:, 0] = root_node # Some predictions did not have the root label as their first value
    # Calculates extent of intersections between predicted labels and valid labels as per the provided graph by cell
    in_graph = np.isin(pred_h_labels, graph.nodes)
    n_valid_labels = np.sum(in_graph, axis=1)
    if verbose:
        invalid_labels = np.unique(
            pred_h_labels[~np.isin(pred_h_labels, graph.nodes)]
        ).tolist()
        print(f'The hierarchical annotations contained {len(invalid_labels)} invalid labels:\n{invalid_labels}.\nThis is only problematic if these are labels you intended to be counted as valid.')
        
    # The last valid label per cell is at n_valid_labels - 1 assuming valid labels start at index 0
    idx_last_valid_label = np.fmax(
        n_valid_labels - 1,
        np.zeros(shape=n_valid_labels.shape)
    ).astype(int)
    pred_labels_flat = np.take_along_axis(
        pred_h_labels,
        idx_last_valid_label[:, np.newaxis],
        axis = 1
    )
    pred_labels_flat = np.squeeze(pred_labels_flat)

    return pred_labels_flat
    
class Hierarchical_Metric():
    def __init__(self, true_labels, predicted_labels, hierarchy_structure, root_node='Blood'):
        '''hierarchy_structure: NetworkX graph of hierarchical classifier'''
        self.true_labels = np.array(true_labels) 
        self.predicted_labels = np.array(predicted_labels) 
        self.hierarchy_structure = hierarchy_structure
        self.root_node = root_node
        self.augmented_lookups = {}
        self.intersect_lookups = {}

    def augmented_set_of_node_n(self, node):
        '''Assuming a tree hierarchy structure, ancestors of node n, including node, excluding root'''
        
        if node not in self.hierarchy_structure.nodes:
            node = self.root_node

        if node not in self.augmented_lookups.keys(): # avoid having to call nx ancestors for every single true and predicted label
            ancestors = nx.ancestors(self.hierarchy_structure, node)
            ancestors.add(node)
            self.augmented_lookups[node] = np.array(list(ancestors))

        return self.augmented_lookups[node]

    def calculate_intersects(self, t_label, p_label, t_label_augmented, p_label_augmented):
        cardinality_intersect_t_p = len(np.intersect1d(t_label_augmented, p_label_augmented))

        #test for over specialization and in case cut augmented p to len of augmented true
        if len(np.intersect1d(t_label_augmented, p_label_augmented)) == len(t_label_augmented):
            over_spec_len = len(p_label_augmented) - len(t_label_augmented)
            slice_len = len(p_label_augmented) - over_spec_len
            p_label_augmented = p_label_augmented[:slice_len]

        cardinality_p_label_augmented = len(p_label_augmented)
        if t_label not in self.intersect_lookups.keys():
            self.intersect_lookups[t_label] = {}
            
        self.intersect_lookups[t_label][p_label] = (cardinality_intersect_t_p, cardinality_p_label_augmented)

    def hP(self, true_labels, predicted_labels):
        numerator = []
        denominator = []
        for t_label, p_label in zip(true_labels, predicted_labels):
            if not (t_label in self.intersect_lookups.keys() and p_label in self.intersect_lookups[t_label].keys()):
                t_label_augmented = self.augmented_set_of_node_n(t_label)
                p_label_augmented = self.augmented_set_of_node_n(p_label) 
                self.calculate_intersects(t_label, p_label, t_label_augmented, p_label_augmented)                

            cardinality_intersect_t_p, cardinality_p_label_augmented = self.intersect_lookups[t_label][p_label]
            numerator.append(cardinality_intersect_t_p)
            denominator.append(cardinality_p_label_augmented)
        
        return np.sum(np.array(numerator)) / np.sum(np.array(denominator))


    def hR(self, true_labels, predicted_labels):
        numerator = []
        denominator = []
        for t_label, p_label in zip(true_labels, predicted_labels):
            t_label_augmented = self.augmented_set_of_node_n(t_label)
            p_label_augmented = self.augmented_set_of_node_n(p_label)
            if not (t_label in self.intersect_lookups.keys() and p_label in self.intersect_lookups[t_label].keys()):
                self.calculate_intersects(t_label, p_label, t_label_augmented, p_label_augmented)  

            cardinality_intersect_t_p, _ = self.intersect_lookups[t_label][p_label]
            cardinality_t_label_augmented = len(t_label_augmented)

            numerator.append(cardinality_intersect_t_p)
            denominator.append(cardinality_t_label_augmented)
        
        return np.sum(np.array(numerator)) / np.sum(np.array(denominator)) 

    def hF(self, beta):

        hP = self.hP(self.true_labels, self.predicted_labels) 
        hR = self.hR(self.true_labels, self.predicted_labels)
        
        hF = (beta**2 + 1) * hP * hR / (beta**2 * hP + hR)

        return hF

    def macro_hF(self, beta):
        '''Macro averaged hF-Score (average of micro hF1's for each label)'''

        labels = pd.Series(self.true_labels).value_counts().keys()
        
        label_Fb = []

        for label in labels: 
            true_label_idcs = np.where(self.true_labels == label)[0]
            hP = self.hP(self.true_labels[true_label_idcs], self.predicted_labels[true_label_idcs])
            hR = self.hR(self.true_labels[true_label_idcs], self.predicted_labels[true_label_idcs])

            Fb = (beta**2 + 1) * hP * hR / (beta**2 * hP + hR)
            label_Fb.append(Fb)

        return np.sum(np.array(label_Fb))/len(labels)

    def list_micro_metrics(self, beta):
        label_metrics = pd.DataFrame(columns=[f'hF{beta}', 'hR', 'hP'])
        for label in np.unique(self.true_labels): 
            true_label_idcs = np.where(self.true_labels == label)[0]
            hP = self.hP(self.true_labels[true_label_idcs], self.predicted_labels[true_label_idcs])
            hR = self.hR(self.true_labels[true_label_idcs], self.predicted_labels[true_label_idcs])

            Fb = (beta**2 + 1) * hP * hR / (beta**2 * hP + hR)
            label_metrics.loc[label] = [np.round(Fb, 2), np.round(hR, 2), np.round(hP, 2)]

        return label_metrics
    
    
   
def annotate_hierarchical(adata, graph, annotate_from_obs_key, root):
    len(adata.obs_names)
    for cell_number, cell_name in enumerate(adata.obs[annotate_from_obs_key]):
        ancestors = nx.shortest_path(graph, source = root, target = cell_name)
        for i,a in enumerate(ancestors):
            try:
                temp_vec = list(adata.obs[f'Level_{i}'])
            except KeyError:
                adata.obs[f'Level_{i}'] = np.nan
                temp_vec = list(adata.obs[f'Level_{i}'])
            temp_vec[cell_number] = a
            adata.obs[f'Level_{i}'] = temp_vec
