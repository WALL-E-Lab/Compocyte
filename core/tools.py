import numpy as np
import tensorflow.keras as keras
from scipy.sparse.csr import csr_matrix
import networkx as nx
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import matplotlib

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

            obs_df = obs_df[obs_df[true_key].isin([np.nan, '', 'nan']) != True]
            obs_df.rename(columns={true_key: 'true_last'}, inplace=True)
            if not true_only:
                obs_df = obs_df[obs_df[pred_key].isin([np.nan, '', 'nan']) != True]
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
            obs_df_level_true = obs_df_level[obs_df_level['true_last'].isin([np.nan, '', 'nan']) != True]           
            level_barcodes_true = [x for x in obs_df_level_true.index if x in obs_df.index]
            obs_df.loc[level_barcodes_true, 'true_last'] = obs_df_level_true.loc[level_barcodes_true, 'true_last']
            if not true_only:
                obs_df_level_pred = obs_df_level[obs_df_level['pred_last'].isin([np.nan, '', 'nan']) != True] 
                level_barcodes_pred = [x for x in obs_df_level_pred.index if x in obs_df.index]
                obs_df.loc[level_barcodes_pred, 'pred_last'] = obs_df_level_pred.loc[level_barcodes_pred, 'pred_last']

    return obs_df

def weighted_accuracy(dict_of_cell_relations, adata, obs_names, test_barcodes, value='pct', is_flat=False, graph=None):
    """Implement accuracy metric that takes into account the distance between predicted and true label.
    Over-specialization errors are not penalized as they are, in this case, not really errors. The last known
    true label is predicted correctly and whatever prediction is made beyond that can not be verified.
    """

    if graph is None:
        graph = nx.DiGraph()
        make_graph_from_edges(dict_of_cell_relations, graph)

    root_node = list(dict_of_cell_relations.keys())[0]
    adata = adata[test_barcodes, :].copy()
    if is_flat:
        if type(obs_names) == list and len(obs_names) > 1:
            raise Exception()

        elif type(obs_names) == list:
            obs_names = obs_names[0]

        last_annotation_df = adata.obs.loc[:, [obs_names, f'{obs_names}_pred']]
        last_annotation_df.rename(columns={obs_names: 'true_last', f'{obs_names}_pred': 'pred_last'}, inplace=True)

    else:
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

def plot_hierarchy_confusions(hierarchy, adata, graph, obs_names, fig_size=(12, 12)):
    con_mat = con_mat_leaf_nodes(hierarchy, adata, graph, obs_names, plot=False)
    graph_weights = graph.copy()
    # Get each confusion affecting more than 20 % of cells truly belonging to a given cell type
    # Confusions across all levels and trees of the hierarchy
    hierarchy_list = []
    generate_node_level_list(hierarchy, hierarchy_list)
    for confusion in np.dstack(np.where(con_mat > 0.2))[0]:
        true_label = hierarchy_list[confusion[0]][0]
        pred_label = hierarchy_list[confusion[1]][0]
        # If confusions are unexpected, i. e. the predicted label is not an ancestor or descendant
        # of the true label, add an edge with the adequate weight to the graph for plotting
        # Replaces multiple confusion matrices as more straight forward and more condensed way
        # to get an overview of relevant confusions
        ancestor_marked = False
        # Ensure that only one connection is drawn to the earliest relevant misclassification
        # in the tree
        for ancestor in nx.shortest_path(graph, 'Blood', pred_label):
            if graph_weights.has_edge(true_label, ancestor):
                ancestor_marked = True
                break

        if not is_pred_parent_or_child_or_equal(graph, true_label, pred_label, 'Blood') and not ancestor_marked:
            graph_weights.add_edge(true_label, pred_label, weight=round(con_mat[confusion[0], confusion[1]] * 100, 1))

    edges_hierarchy = list(graph.edges())
    edges_confusion = [e for e in graph_weights.edges() if not e in edges_hierarchy]
    fig, ax = plt.subplots(figsize=fig_size)
    pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='twopi')
    nx.draw(graph_weights, pos, with_labels=True, arrows=True, ax=ax, edgelist=[])
    nx.draw_networkx_edges(graph_weights, pos, edgelist=edges_hierarchy, width=1.5, style='dashed')
    labels = nx.get_edge_attributes(graph_weights, 'weight')
    for l in labels.keys():
        nx.draw_networkx_edges(graph_weights, pos, edgelist=[l], width=2, edge_color=matplotlib.cm.get_cmap('viridis')(labels[l]/100)[:-1], arrowsize=10, connectionstyle='arc3,rad=0.4', label='test')
    for l in labels.keys():
        labels[l] = f'{labels[l]} %'

    sm = plt.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('viridis'), norm=plt.Normalize(vmin = 0, vmax=100))
    sm._A = []
    # plt.colorbar(sm, label='% of cells misclassified', orientation='horizontal', fraction=0.046, pad=0.04) perfect match
    plt.colorbar(sm, label='% of cells misclassified', orientation='horizontal', fraction=0.03, pad=0.04)
    plt.title(
        'Confusions of > 20 % of true cells, visualized as directed edge to the earliest misstep in the hierarchical classification process',
        fontdict={'fontsize': 'x-large'}
    )

def last_level_con_mat(dict_of_cell_relations, adata, obs_name, labels, graph=None):
    if graph is None:
        graph = nx.DiGraph()
        make_graph_from_edges(dict_of_cell_relations, graph)

    con_mat = confusion_matrix(adata.obs[obs_name], adata.obs[f'{obs_name}_pred'], labels=labels, normalize='true')
    con_mat_disp = ConfusionMatrixDisplay(con_mat, display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 12))
    con_mat_disp.plot(xticks_rotation='vertical', ax=ax, values_format='.2f')

    return con_mat