import numpy as np
import tensorflow.keras as keras

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