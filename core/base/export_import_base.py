from copy import deepcopy
from classiFire.core.models.dense import DenseKeras
from classiFire.core.models.dense_torch import DenseTorch
import networkx as nx
from tensorflow import keras
import torch
import sklearn

class ExportImportBase():
    def make_classifier_dict(self, node):
        """Export local classifier into externally applicable format"""

        classifier_dict = {}
        classifier_dict['classifier'] = self.graph.nodes[node]['local_classifier']
        classifier_dict['data_type'] = self.graph.nodes[node]['local_classifier'].data_type
        if 'label_encoding' in self.graph.nodes[node]:
            classifier_dict['label_encoding'] = self.graph.nodes[node]['label_encoding']

        if 'selected_var_names' in self.graph.nodes[node]:
            classifier_dict['selected_var_names'] = self.graph.nodes[node]['selected_var_names']

        else:
            pass
            # Upon reconstruction simply use all of data type as input

        return classifier_dict

    def export_classifiers(self):
        dict_of_cell_relations = deepcopy(self.dict_of_cell_relations)
        for node in list(self.graph):
            if 'local_classifier' in self.graph.nodes[node]:
                if type(self.graph.nodes[node]['local_classifier']) not in [DenseKeras, DenseTorch]:
                    raise Exception('Classifier exported not currently implemented for this type of classifier.')

                path_to_node = nx.shortest_path(self.graph, self.root_node, node)
                child_dict = dict_of_cell_relations
                for p in path_to_node:
                    child_dict = child_dict[p]

                child_dict['classifier'] = self.make_classifier_dict(node)

        return dict_of_cell_relations

    def import_classifier(self, node, classifier_dict, overwrite=False):
        for a in ['classifier', 'label_encoding', 'data_type', 'selected_var_names']:
            if not a in classifier_dict:
                raise KeyError(f'Missing key {a} for successful classifier import.')

        classifier_exists = 'local_classifier' in self.graph.nodes[node]
        if overwrite:
            if type(classifier_dict['classifier']) in [DenseKeras, DenseTorch]:
                self.graph.nodes[node]['local_classifier'] = classifier_dict['classifier']

            elif issubclass(classifier_dict['classifier'], torch.nn.Module):
                if not 'fit_function' in classifier_dict or not 'predict_function' in classifier_dict:
                    raise KeyError(f'Missing key fit_function/predict_function for successful classifier import.')

                self.graph.nodes[node]['local_classifier'] = DenseTorch.import_external(
                    model=classifier_dict['classifier'],
                    data_type=classifier_dict['data_type'],
                    fit_function=classifier_dict['fit_function'],
                    predict_function=classifier_dict['predict_function']
                )

            elif issubclass(classifier_dict['classifier'], keras.Model):
                self.graph.nodes[node]['local_classifier'] = DenseKeras.import_external(
                    classifier_dict['classifier'], 
                    classifier_dict['data_type'])
                
            else:
                raise Exception('Cannot currently import classifier of this type. Please post an issue on Github.')

            for k in ['label_encoding', 'data_type', 'selected_var_names']:
                self.graph.nodes[node][k] = classifier_dict[k]

        else:
            print(f'Classifier already exists at {node} and overwrite is set to False.')

    def import_classifiers(self, dictionary, overwrite=False, parent_key=None):
        for key in dictionary.keys():
            if key == 'classifier':
                import_classifier(parent_key, dictionary[key], overwrite=overwrite)
            
            elif type(dictionary[key]) == dict and len(dictionary[key].keys()) > 0:
                import_classifiers(dictionary[key], overwrite=True, parent_key=key)