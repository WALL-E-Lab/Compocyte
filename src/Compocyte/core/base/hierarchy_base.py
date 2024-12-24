import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Compocyte.core.tools import flatten_dict, dict_depth, hierarchy_names_unique, \
    make_graph_from_edges, set_node_to_depth, delete_dict_entries
from Compocyte.core.models.log_reg import LogisticRegression
from copy import deepcopy

class HierarchyBase():
    """Add explanation
    """

    def set_cell_relations(self, root_node, dict_of_cell_relations, obs_names):
        """Once set, cell relations can only be changed one node at a time, using supplied methods,
        not by simply calling defining new cell relations
        """

        if self.root_node is not None and self.dict_of_cell_relations is not None and self.obs_names is not None:
            raise Exception('To redefine cell relations after initialization, call update_hierarchy.')

        dict_of_cell_relations_with_classifiers = deepcopy(dict_of_cell_relations)
        dict_of_cell_relations, contains_classifier = delete_dict_entries(dict_of_cell_relations, 'classifier')
        self.root_node = root_node
        self.ensure_depth_match(dict_of_cell_relations, obs_names)
        self.ensure_unique_nodes(dict_of_cell_relations)
        self.dict_of_cell_relations = dict_of_cell_relations
        self.obs_names = obs_names
        self.all_nodes = flatten_dict(self.dict_of_cell_relations)
        self.node_to_depth = set_node_to_depth(self.dict_of_cell_relations)
        self.make_classifier_graph()        
        if contains_classifier:
            self.import_classifiers(dict_of_cell_relations_with_classifiers)

    def ensure_depth_match(self, dict_of_cell_relations, obs_names):
        """Check if the annotations supplied in .obs under obs_names are sufficiently deep to work 
        with the hierarchy provided.
        """

        if not dict_depth(dict_of_cell_relations) == len(obs_names):
            raise Exception('obs_names must contain an annotation key for every level of the '\
                'hierarchy supplied in dict_of_cell_relations.')

    def ensure_unique_nodes(self, dict_of_cell_relations):
        """Check if keys within the hierarchy are unique across all levels as that is a requirement
        for uniquely identifying graph nodes with networkx.
        """

        if not hierarchy_names_unique(dict_of_cell_relations):
            raise Exception('Names given in the hierarchy must be unique.')

    def make_classifier_graph(self):
        """Compute directed graph from a given dictionary of cell relationships."""

        self.graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, self.graph)

    def plot_hierarchy(self):
        """Plot hierarchical cell labels.
        """

        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='twopi')
        nx.draw(self.graph, pos, with_labels=True, arrows=True, ax=ax)

    def get_children_obs_key(self, parent_node):
        """Get the obs key under which labels for the following level in the hierarchy are saved.
        E. g. if you get_children_obs_key for T cells, it will return the obs key for alpha beta
        T cell labels and so on.
        """

        depth_parent = self.node_to_depth[parent_node]
        children_obs_key = self.obs_names[depth_parent + 1]

        return children_obs_key

    def get_parent_obs_key(self, parent_node):
        """Get the obs key under which labels for the current level in the hierarchy are saved.
        E. g. if you get_parent_obs_key for T cells, it will return the obs key in which true
        T cells are labelled as such.
        """

        depth_parent = self.node_to_depth[parent_node]

        return self.obs_names[depth_parent]
    
    def get_child_nodes(self, node):
        return self.graph.adj[node].keys()

    def get_leaf_nodes(self):
        return [
            x for x in self.graph.nodes() \
            if self.graph.out_degree(x) == 0 \
            and self.graph.in_degree(x) == 1
        ]

    def get_parent_node(self, node, graph=None):
        if graph is None:
            graph = self.graph

        edges = np.array(graph.edges)
        # In a directed graph there should only be one edge leading TO any given node
        idx_child_node_edges = np.where(edges[:, 1] == node)
        parent_node = edges[idx_child_node_edges][0, 0]

        return parent_node

    def update_hierarchy(self, dict_of_cell_relations, root_node=None, overwrite=False):
        dict_of_cell_relations_with_classifiers = deepcopy(dict_of_cell_relations)
        dict_of_cell_relations, contains_classifier = delete_dict_entries(dict_of_cell_relations, 'classifier')
        if root_node is not None:
            self.root_node = root_node

        if dict_of_cell_relations == self.dict_of_cell_relations:
            return

        self.ensure_depth_match(dict_of_cell_relations, self.obs_names)
        self.ensure_unique_nodes(dict_of_cell_relations)
        all_nodes_pre = flatten_dict(self.dict_of_cell_relations)
        self.dict_of_cell_relations = dict_of_cell_relations
        all_nodes_post = flatten_dict(self.dict_of_cell_relations)
        self.all_nodes = all_nodes_post
        self.node_to_depth = set_node_to_depth(self.dict_of_cell_relations)
        new_graph = nx.DiGraph()
        make_graph_from_edges(self.dict_of_cell_relations, new_graph)

        new_nodes = [n for n in all_nodes_post if n not in all_nodes_pre]
        [n for n in all_nodes_pre if n not in all_nodes_post]
        moved_nodes = []
        classifier_nodes = []
        for node in all_nodes_post:
            if node in new_nodes:
                continue

            # Check if node was moved within the hierarchy, i. e. assigned
            # to a different parent node
            # Does not change the strategy of assigning the previous node attributes
            # but may end up a fact of interest
            if not node == self.root_node:
                parent_post = self.get_parent_node(node, graph=new_graph)
                parent_pre = self.get_parent_node(node)
                if parent_pre != parent_post:
                    moved_nodes.append(node)

            # Transfer properties, such as local classifier, from old graph
            # to new graph
            for item in self.graph.nodes[node]:
                new_graph.nodes[node][item] = deepcopy(self.graph.nodes[node][item])

            if "local_classifier" in self.graph.nodes[node]:
                classifier_nodes.append(node) # Define nodes that contain a classifier

            print(f'Transfered to {node}, local classifier {"transferred" if "local_classifier" in self.graph.nodes[node] else "not transferred"}')

        self.graph = new_graph
        for node in classifier_nodes:
            print(f'Ensuring correct output architecture for {node}.')
            child_nodes = self.get_child_nodes(node)
            # Previously reset all classifier nodes
            # Bad idea because you want to conserve as much of the training progress as possible,
            # resetting as little as possible, as much as necessary
            if True in [n in new_nodes or n in moved_nodes for n in [node] + list(child_nodes)]:
                if type(self.graph.nodes[node]['local_classifier']) is LogisticRegression:
                    print('Cannot adjust output structure of logistic regression classifier.')
                    continue

                # reset label encoding, unproblematic because the final layer is reinitilaized anyway
                self.graph.nodes[node]['label_encoding'] = {}
                self.graph.nodes[node]['local_classifier'].reset_output(len(child_nodes))

        if contains_classifier:
            self.import_classifiers(dict_of_cell_relations_with_classifiers, overwrite=overwrite)