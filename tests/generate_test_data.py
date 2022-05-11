import numpy as np
import scanpy as sc
import pandas as pd
import random
import string
from scipy import sparse

def generate_test_h5ad(hierarchy, output_file='', n_genes=20000, n_cells=2000, min_counts=0, max_counts=80):
    # Generate an expression matrix containing counts randomly chosen from integers min_counts to max_counts
    expression_matrix = np.random.randint(
        min_counts, 
        max_counts + 1, 
        (n_cells, n_genes))
    gene_names = ['A', 'A']
    # Ensure that the list of gene names generated is unique
    while len(gene_names) != len(set(gene_names)):
        # Generate gene names consisting of 3 uppercase letters trailed by 2 digits
        gene_names = [
            ''.join(
                [random.choice(group) for group in [
                    string.ascii_uppercase,
                    string.ascii_uppercase,
                    string.ascii_uppercase,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                ]]) for _ in range(n_genes)
            ]

    barcodes = ['A', 'A']
    while len(barcodes) != len(set(barcodes)):
        # Generate barcodes consisting of 3 uppercase letters trailed by 4 digits
        barcodes = [
            ''.join(
                [random.choice(group) for group in [
                    string.ascii_uppercase,
                    string.ascii_uppercase,
                    string.ascii_uppercase,
                    string.ascii_uppercase,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                    string.digits,
                ]]) for _ in range(n_cells)
            ]

    test_adata = sc.AnnData(
        sparse.csr_matrix(expression_matrix),
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names))
    if output_file != '':
        test_adata.write_h5ad(output_file)
        
    return test_adata