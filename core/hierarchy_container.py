import networkx as nx
from .tools import flatten_dict, dict_depth, hierarchy_names_unique, make_graph_from_edges, \
	set_node_to_depth

class HierarchyContainer():
	"""Add explanation
	"""

	def __init__(self, dict_of_cell_relations, obs_names):
		"""Add explanation
		"""

		self.dict_of_cell_relations = dict_of_cell_relations
        self.obs_names = obs_names

        # Basic validation
        self.ensure_depth_match()
        self.ensure_unique_nodes()

		self.all_nodes = flatten_dict(self.dict_of_cell_relations)
		self.node_to_depth = set_node_to_depth(self.dict_of_cell_relations)
        self.node_to_scVI = set_node_to_scVI(self.dict_of_cell_relations)
		self.make_classifier_graph()

    def ensure_depth_match(self):
    	"""Check if the annotations supplied in .obs under obs_names are sufficiently deep to work 
    	with the hierarchy provided.
    	"""

    	if not dict_depth(self.dict_of_cell_relations) == len(self.obs_names):
            raise Exception('obs_names must contain an annotation key for every level of the '\
            	'hierarchy supplied in dict_of_cell_relations.')

    def ensure_unique_nodes(self):
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

    	pos = nx.drawing.nx_agraph.graphviz_layout(self.graph, prog='dot')
		nx.draw(self.graph, pos, with_labels=True, arrows=False)

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