import os
import pytest
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.models.trees import BoostedTrees
from Compocyte.core.tools import infer_levels
from Compocyte.data import sample_data, sample_hierarchy

@pytest.fixture(scope="session")
def shared_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("shared")

def test_training(shared_tmp):
    adata = sample_data()
    hierarchy = sample_hierarchy()
    _, obs_names = infer_levels(
        hierarchy=hierarchy, 
        labels='labels', 
        root_node='T_cell', 
        adata=adata)
    hc = HierarchicalClassifier(
        save_path=os.path.join(shared_tmp, 'test_model'),
        adata=adata,
        root_node='T_cell',
        dict_of_cell_relations=hierarchy,
        obs_names=obs_names)
    hc.set_classifier_type('CD4-T', BoostedTrees)
    hc.set_classifier_type('CD8-T', LogisticRegression)
    hc.train_all_child_nodes()
    hc.save()
    assert os.path.exists(os.path.join(shared_tmp, 'test_model', 'models'))
    assert 'local_classifier' in hc.graph.nodes['T_cell']
    assert 'selected_var_names' in hc.graph.nodes['T_cell']
    assert isinstance(hc.graph.nodes['T_cell']['local_classifier'], DenseTorch)
    assert isinstance(hc.graph.nodes['CD4-T']['local_classifier'], BoostedTrees)
    assert isinstance(hc.graph.nodes['CD8-T']['local_classifier'], LogisticRegression)
    assert hc.graph.nodes['T_cell']['local_classifier'].is_fitted
    assert hc.graph.nodes['CD4-T']['local_classifier'].is_fitted
    assert hc.graph.nodes['CD8-T']['local_classifier'].is_fitted

def test_loading(shared_tmp):
    hc = HierarchicalClassifier(os.path.join(shared_tmp, 'test_model'))
    hc.load()
    assert 'local_classifier' in hc.graph.nodes['T_cell']
    assert 'selected_var_names' in hc.graph.nodes['T_cell']
    assert isinstance(hc.graph.nodes['T_cell']['local_classifier'], DenseTorch)
    assert isinstance(hc.graph.nodes['CD4-T']['local_classifier'], BoostedTrees)
    assert isinstance(hc.graph.nodes['CD8-T']['local_classifier'], LogisticRegression)