from copy import deepcopy
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.log_reg import LogisticRegression
import networkx as nx
import torch
import sklearn

from Compocyte.core.models.trees import BoostedTrees

class ExportImportBase():
    def make_classifier_dict(self, node):
        """Export local classifier into externally applicable format"""

        classifier_dict = {}
        classifier_dict['classifier'] = self.graph.nodes[node]['local_classifier']
        classifier_dict['data_type'] = self.default_input_data
        if 'selected_var_names' in self.graph.nodes[node]:
            classifier_dict['selected_var_names'] = self.graph.nodes[node]['selected_var_names']

        else:
            classifier_dict['selected_var_names'] = list(self.var_names)

        return classifier_dict

    def export_classifiers(self):
        dict_of_cell_relations = deepcopy(self.dict_of_cell_relations)
        for node in list(self.graph):
            if 'local_classifier' in self.graph.nodes[node]:
                if type(self.graph.nodes[node]['local_classifier']) not in [DenseTorch, LogisticRegression, BoostedTrees, DummyClassifier]:
                    raise Exception('Classifier exported not currently implemented for this type of classifier.')

                path_to_node = nx.shortest_path(self.graph, self.root_node, node)
                child_dict = dict_of_cell_relations
                for p in path_to_node:
                    child_dict = child_dict[p]

                child_dict['classifier'] = self.make_classifier_dict(node)

        return dict_of_cell_relations

    def import_classifier(self, node, classifier_dict, overwrite=False):
        for a in ['classifier', 'data_type', 'selected_var_names']:
            if a not in classifier_dict:
                raise KeyError(f'Missing key {a} for successful classifier import.')

        if classifier_dict['data_type'] != self.default_input_data:
            raise Exception('Data type of supplied classifier must match other local classifiers: {self.default_input_data}')
        
        classifier_exists = 'local_classifier' in self.graph.nodes[node]
        if (classifier_exists and overwrite) or not classifier_exists:
            if type(classifier_dict['classifier']) in [DenseTorch, LogisticRegression, DummyClassifier, BoostedTrees]:
                self.graph.nodes[node]['local_classifier'] = classifier_dict['classifier']

            elif issubclass(type(classifier_dict['classifier']), torch.nn.Module):
                if 'fit_function' not in classifier_dict or 'predict_function' not in classifier_dict:
                    raise KeyError('Missing key fit_function/predict_function for successful classifier import.')

                self.graph.nodes[node]['local_classifier'] = DenseTorch.import_external(
                    model=classifier_dict['classifier'],
                    data_type=classifier_dict['data_type'],
                    fit_function=classifier_dict['fit_function'],
                    predict_function=classifier_dict['predict_function']
                )

            elif issubclass(type(classifier_dict['classifier']), sklearn.linear_model.LogisticRegression):
                self.graph.nodes[node]['local_classifier'] = LogisticRegression(
                    classifier_dict['classifier'], 
                    classifier_dict['data_type'])
                
            else:
                raise Exception('Cannot currently import classifier of this type. Please post an issue on Github.')

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

    def import_classifiers(self, dictionary, parent_key, overwrite=False):
        for key in dictionary.keys():
            if key == 'classifier':
                self.import_classifier(parent_key, dictionary[key], overwrite=overwrite)
            
            elif isinstance(dictionary[key], dict) and len(dictionary[key].keys()) > 0:
                self.import_classifiers(dictionary[key], overwrite=True, parent_key=key)