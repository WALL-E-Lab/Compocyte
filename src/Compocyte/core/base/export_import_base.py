from copy import deepcopy
import os
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.log_reg import LogisticRegression
import networkx as nx
import torch
import sklearn

from Compocyte.core.models.trees import BoostedTrees

class ExportImportBase():
    def make_classifier_dict(self, node, temp_path):
        """Export local classifier into externally applicable format"""

        classifier_dict = {}
        classifier_dict['classifier'] = os.path.join(temp_path, node)
        if not os.path.exists(os.path.join(temp_path, node)):
            os.makedirs(os.path.join(temp_path, node))
            
        self.graph.nodes[node]['local_classifier']._save(os.path.join(temp_path, node))
        
        classifier_dict['data_type'] = self.default_input_data
        if 'selected_var_names' in self.graph.nodes[node]:
            classifier_dict['selected_var_names'] = self.graph.nodes[node]['selected_var_names']

        else:
            classifier_dict['selected_var_names'] = list(self.var_names)

        return classifier_dict

    def export_classifiers(self, temp_path):
        dict_of_cell_relations = deepcopy(self.dict_of_cell_relations)
        for node in list(self.graph):
            if 'local_classifier' in self.graph.nodes[node]:
                if type(self.graph.nodes[node]['local_classifier']) not in [DenseTorch, LogisticRegression, BoostedTrees, DummyClassifier]:
                    raise Exception('Classifier exported not currently implemented for this type of classifier.')

                path_to_node = nx.shortest_path(self.graph, self.root_node, node)
                child_dict = dict_of_cell_relations
                for p in path_to_node:
                    child_dict = child_dict[p]

                child_dict['classifier'] = self.make_classifier_dict(node, temp_path=temp_path)

        return dict_of_cell_relations

    def import_classifier(self, node, classifier_dict, temp_path, overwrite=False):
        for a in ['classifier', 'data_type', 'selected_var_names']:
            if a not in classifier_dict:
                raise KeyError(f'Missing key {a} for successful classifier import.')

        if classifier_dict['data_type'] != self.default_input_data:
            raise Exception('Data type of supplied classifier must match other local classifiers: {self.default_input_data}')
        
        classifier_exists = 'local_classifier' in self.graph.nodes[node]
        if (classifier_exists and overwrite) or not classifier_exists:
            contents = os.listdir(os.path.join(temp_path, node))
            if len([c for c in contents if c.startswith('non_param_dict')]) > 0:
                    classifier = DenseTorch._load(os.path.join(temp_path, node))

            elif 'labels_dec.pickle' in contents and not 'model.cbm' in contents:
                classifier = LogisticRegression._load(os.path.join(temp_path, node))

            elif 'labels_dec.pickle' in contents and 'model.cbm' in contents:
                classifier = BoostedTrees._load(os.path.join(temp_path, node))

            else:
                classifier = DummyClassifier._load(os.path.join(temp_path, node))

            self.graph.nodes[node]['local_classifier'] = classifier
            sel_var = classifier_dict['selected_var_names']
            var_not_present = [v for v in sel_var if v not in self.var_names]
            if len(var_not_present) > 0:
                self.var_names = self.var_names + var_not_present
                if self.adata is not None:
                    self.add_variables(var_not_present)

            for k in ['label_encoding', 'selected_var_names']:
                if k in classifier_dict:
                    self.graph.nodes[node][k] = classifier_dict[k]

        else:
            print(f'Classifier already exists at {node} and overwrite is set to False.')

    def import_classifiers(self, dictionary, parent_key, temp_path, overwrite=False):
        for key in dictionary.keys():
            if key == 'classifier':
                self.import_classifier(parent_key, dictionary[key], temp_path, overwrite=overwrite)
            
            elif isinstance(dictionary[key], dict) and len(dictionary[key].keys()) > 0:
                self.import_classifiers(dictionary[key], temp_path=temp_path, overwrite=True, parent_key=key)