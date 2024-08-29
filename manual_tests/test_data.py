import networkx as nx
import numpy as np
import scanpy as sc
import pandas as pd
from scipy import sparse
from Compocyte.core.tools import get_leaf_nodes, make_graph_from_edges, \
    dict_depth


def get_hierarchy():
    hierarchy = {
        'Blood': {
            'L': {
                'TNK': {
                    'T': {},
                    'NKT': {},
                    'NK': {},
                    'ILC': {}
                },
                'BP': {},
                'M': {}
            },
            'NL': {}
        }
    }

    return hierarchy


def generate_test_adata(cells_per_leaf_node=50, n_genes=100):
    hierarchy = get_hierarchy()
    graph = nx.DiGraph()
    root_node = list(hierarchy.keys())[0]
    make_graph_from_edges(hierarchy, graph)
    leaf_nodes = get_leaf_nodes(hierarchy)
    obs_names = [f'Level_{i}' for i in range(dict_depth(hierarchy))]
    total_cells = cells_per_leaf_node * len(leaf_nodes)
    count_matrix = sparse.csr_matrix(
        np.random.randint(0, 40, (total_cells, n_genes))
    )
    var = pd.DataFrame(
        index=[f'Gene_{i}' for i in range(n_genes)]
    )
    obs = pd.DataFrame()
    obs['barcodes'] = [f'Cell_{i}' for i in range(total_cells)]
    adata = sc.AnnData(
        X=count_matrix,
        var=var,
        obs=obs
    )
    adata.obs.set_index('barcodes', inplace=True)
    batch_names = np.array([f'Batch_{i}' for i in range(3)])
    adata.obs['batch'] = np.random.choice(batch_names, (len(adata)))
    for i, node in enumerate(leaf_nodes):
        path_to_node = nx.shortest_path(graph, root_node, node)
        while len(path_to_node) < len(obs_names):
            path_to_node.append('')

        starting_index = i * cells_per_leaf_node
        ending_index = min(len(adata), (i + 1) * cells_per_leaf_node)
        adata.obs.loc[
            adata.obs_names[starting_index:ending_index],
            obs_names] = path_to_node

    return adata
