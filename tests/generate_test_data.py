import numpy as np
import scanpy as sc
import pandas as pd
import random
import string
from scipy import sparse
from math import floor

test_hierarchy = {'L': {'B_PB': {'B': {'DN_B': {}, 'Memory_B': {}, 'Naive_B': {}}, 'PB': {'Non_proliferating_PB': {}, 'Proliferating_PB': {}}},
'M': {'DC': {'ADC': {}, 'MoDC': {}, 'cDC': {}, 'pDC': {}}, 'GR': {}, 'MO': {'cMO': {}, 'iMO': {}, 'ncMO': {}}},
'T_ILC': {'ILC_ALL': {'ILC': {}, 'NK': {}}, 'T': {'MAIT': {}, 'ab_T': {}, 'gd_T': {}}}}}
test_obs_names = ['Level_1', 'Level_2', 'Level_3', 'Level_4']

def generate_random_names(n, format):
    names = []
    for _ in range(n):
        # Ensure unique names
        while True:
            x = ''.join(
                [random.choice(group) for group in format])
            if x in names:
                continue

            else:
                names.append(x)
                break

    return names

def get_leaf_nodes(hierarchy):
    leaf_nodes = []
    for node in hierarchy.keys():
        if len(hierarchy[node].keys()) != 0:
            leaf_nodes += get_leaf_nodes(hierarchy[node])

        else:
            leaf_nodes += [node]

    return leaf_nodes

def trace_path(hierarchy, current_node, prior_history=[], node_to_path={}):
    for node in hierarchy.keys():
        if len(hierarchy[node].keys()) != 0:
            new_node_to_path = trace_path(
                hierarchy[node], 
                node,
                prior_history + [current_node])
            for leaf_node in new_node_to_path.keys():
                node_to_path[leaf_node] = new_node_to_path[leaf_node]

        else:
            node_to_path[node] = prior_history + [current_node] + [node]

    return node_to_path

def generate_random_anndata(
    hierarchy=test_hierarchy,
    obs_names=test_obs_names,
    leaf_nodes_mandatory=True, 
    n_genes=20000, 
    n_cells=2000, 
    min_counts=0, 
    max_counts=80,
    batch_key='batch'):

    gene_names = generate_random_names(
        n_genes,
        [
            string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase, 
            string.digits, string.digits, string.digits, string.digits, string.digits, 
            string.digits, string.digits, string.digits, string.digits, string.digits
        ])
    barcodes = generate_random_names(
        n_cells,
        [
            string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase, string.ascii_uppercase,
            string.digits, string.digits, string.digits, string.digits, string.digits, string.digits,
        ])
    expression_matrix = np.random.randint(
        min_counts, 
        max_counts + 1, 
        (n_cells, n_genes))
    test_adata = sc.AnnData(
        sparse.csr_matrix(expression_matrix),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names))
    paths_to_leaf_nodes = trace_path(hierarchy['L'], 'L')
    n_leaf_nodes = len(paths_to_leaf_nodes.keys())
    samples_per_node = n_cells / n_leaf_nodes
    for i, node in enumerate(paths_to_leaf_nodes):
        relevant_barcodes = test_adata.obs_names[floor(i * samples_per_node):floor((i + 1) * samples_per_node)]
        for n, path_element in enumerate(paths_to_leaf_nodes[node]):
            test_adata.obs.loc[relevant_barcodes, obs_names[n]] = path_element

    for obs_name in obs_names:
        test_adata.obs[obs_name] = pd.Categorical(test_adata.obs[obs_name])

    test_adata.obs[batch_key] = 'batch_1'

    return test_adata.copy()