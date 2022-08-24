from copy import deepcopy
from classiFire.core.models.neural_network import NeuralNetwork
import networkx as nx

class ExportImportBase():
    def make_classifier_dict(self, node):
        """Export local classifier into externally applicable format"""

        classifier_dict = {}
        classifier_dict['classifier'] = self.graph.nodes[node]['local_classifier']
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
                if type(self.graph.nodes[node]['local_classifier']) is not NeuralNetwork:
                    raise Exception('Classifier exported not currently implemented for this type of classifier.')

                path_to_node = nx.shortest_path(self.graph, self.root_node, node)
                child_dict = dict_of_cell_relations
                for p in path_to_node:
                    child_dict = child_dict[p]

                child_dict['classifier'] = self.make_classifier_dict(node)

        return dict_of_cell_relations