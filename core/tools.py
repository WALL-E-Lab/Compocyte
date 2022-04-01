import numpy as np
import tensorflow.keras as keras
from scipy.sparse.csr import csr_matrix

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

    return contains_negative_values or contains_non_whole_numbers

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