import numpy as np
import tensorflow.keras as keras
from scipy.sparse.csr import csr_matrix
import networkx as nx

def set_node_to_depth(dictionary, depth=0, node_to_depth={}):
    for node in dictionary.keys():
        node_to_depth = set_node_to_depth(dictionary[node], depth=depth+1)
        node_to_depth[node] = depth

    return node_to_depth

def set_node_to_scVI(dictionary, parent_node=None, depth=0, max_depth_scVI=2, node_to_scVI={}):
    """Create dict assigning to each node in the hierarchy the node whose scVI dimensions should
    be used. The idea is that running scVI separately for the first max_depth_scVI + 1 levels
    will better bring out subtle differences between cells, improving classifier accuracy.
    """

    if len(dictionary.keys()) == 0:
        pass

    else:
        for key in dictionary.keys():
            deeper_assignments = set_node_to_scVI(dictionary[key], parent_node=key, depth=depth+1)
            if depth <= max_depth_scVI:
                node_to_scVI[key] = key

            else:
                node_to_scVI[key] = parent_node

            for deeper_key, value in deeper_assignments.items():
                node_to_scVI[deeper_key] = value

    return node_to_scVI

def is_counts(matrix, n_rows_to_try=100):
    """Determines whether or not a matrix (such as adata.X, adata.raw.X or an adata layer) contains
    count data by manually checking a subsample of the supplied matrix.
    """

    if not (type(matrix) == np.matrix or type(matrix) == csr_matrix):
        raise ValueError(f'Matrix supplied must be of type {csr_matrix} or {np.matrix}.')

    test_data = matrix[:n_rows_to_try]
    if type(test_data) == csr_matrix:
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

def z_transform_properties(data_arr):
    """Calculates a z transformation to center properties across cells in data_arr \
    around mean zero
    """

    mean_vals = np.mean(data_arr, axis=0)
    std_val = np.std(data_arr)
    data_transformed = (data_arr - mean_vals) / std_val

    return data_transformed

# Remove?
def process_y_input_data(y_input, fitted_label_encoder):
    """Add explanation
    """

    y_input_data_int = fitted_label_encoder.transform(y_input)
    y_input_onehot = keras.utils.to_categorical(y_input_data_int)

    return y_input_data_int, y_input_onehot

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

def list_subgraph_nodes(g, parent_node):
    """Add explanation
    """

    list_of_nodes = []
    for node in g.adj[parent_node].keys():
        if len(g.adj[parent_node].keys()) != 0:
            list_of_nodes.append(node)
            list_of_nodes = list_of_nodes + list_subgraph_nodes(g, node)

        else:
            list_of_nodes.append(node)

    return list_of_nodes

def get_last_annotation(obs_names, adata, barcodes=None):
    if barcodes is None:
        barcodes = adata.obs_names
        
    obs_names_pred = [f'{x}_pred' for x in obs_names]
    for i, (true_key, pred_key) in enumerate(zip(obs_names, obs_names_pred)):
        if i == 0:
            obs_df = adata.obs.loc[barcodes, [true_key, pred_key]]
            obs_df = obs_df[obs_df[true_key].isin([np.nan, '', 'nan']) != True]
            obs_df = obs_df[obs_df[pred_key].isin([np.nan, '', 'nan']) != True]
            obs_df.rename(columns={true_key: 'true_last', pred_key: 'pred_last'}, inplace=True)
            obs_df = obs_df.astype(str)     

        else:
            obs_df_level = adata.obs.loc[barcodes, [true_key, pred_key]]
            obs_df_level.rename(columns={true_key: 'true_last', pred_key: 'pred_last'}, inplace=True)   
            obs_df_level = obs_df_level.astype(str)
            obs_df_level_true = obs_df_level[obs_df_level['true_last'].isin([np.nan, '', 'nan']) != True]
            obs_df_level_pred = obs_df_level[obs_df_level['pred_last'].isin([np.nan, '', 'nan']) != True]            
            level_barcodes_true = [x for x in obs_df_level_true.index if x in obs_df.index]
            level_barcodes_pred = [x for x in obs_df_level_pred.index if x in obs_df.index]
            obs_df.loc[level_barcodes_true, 'true_last'] = obs_df_level_true.loc[level_barcodes_true, 'true_last']
            obs_df.loc[level_barcodes_pred, 'pred_last'] = obs_df_level_pred.loc[level_barcodes_pred, 'pred_last']

    return obs_df

def weighted_accuracy(dict_of_cell_relations, adata, graph, obs_names, value='pct', is_flat=False):
    """Implement accuracy metric that takes into account the distance between predicted and true label.
    Over-specialization errors are not penalized as they are, in this case, not really errors. The last known
    true label is predicted correctly and whatever prediction is made beyond that can not be verified.
    """

    root_node = list(dict_of_cell_relations.keys())[0]
    last_annotation_df = get_last_annotation(obs_names, adata)
    graph = graph.to_undirected()
    for true_node in graph.nodes():
        for pred_node in graph.nodes():
            if not ((last_annotation_df['true_last'] == true_node) & (last_annotation_df['pred_last'] == pred_node)).any():
                continue

            shortest_path = nx.shortest_path(graph, true_node, pred_node)
            n_edges_between = len(shortest_path) - 1
            accuracy_weight = 0.5 ** max(n_edges_between, 0)
            # Avoid penalizing over-specialization
            if true_node in nx.shortest_path(graph, root_node, pred_node):
                accuracy_weight = 1

            last_annotation_df.loc[
                (last_annotation_df['true_last'] == true_node) \
                & (last_annotation_df['pred_last'] == pred_node),
                'accuracy_weight'] = accuracy_weight

    weighted_accuracy = np.mean(last_annotation_df['accuracy_weight'])
    if value == 'pct':
        weighted_accuracy = round(weighted_accuracy * 100, 2)

    return weighted_accuracy