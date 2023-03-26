from scFlorist import core
from .generate_test_data import *

adata = generate_random_anndata(n_cells=600)
def test_scvi_res_train_val():
    data_container = core.sequencing_data_container.SequencingDataContainer(adata, 'trained_classifier')
    hierarchy_container = core.hierarchy_container.HierarchyContainer(test_hierarchy, test_obs_names)
    c = core.hierarchical_classifier.HierarchicalClassifier(data_container, hierarchy_container, 'trained_classifier', use_scVI=True)
    c.train_child_nodes_with_validation('L')