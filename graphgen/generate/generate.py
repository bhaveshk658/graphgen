from graphgen.generate.utils import distance, dist_point_to_line
from graphgen.graph.graph import Graph
from graphgen.graph.node import Node

def to_merge(candidate, G, dist_limit, heading_limit):
	'''
	Determines if a candidate node should be merged into the graph.
	Returns whether or not to merge, the target edge,
	the closest node, and the distance.
	'''
	if len(G.edges()) == 0:
		return False, None

	edges = G.edges()

	# Find edges that satisfy merge conditions.
	for edge in edges:
		temp_dist = dist_point_to_line(edge[0], edge[1], candidate)
		temp_heading = abs(candidate.heading - edge[0].heading)
		
		# Check merge parameters.
		if temp_dist < dist_limit and temp_heading < heading_limit:
			d1 = distance(edge[0].x, edge[0].y, candidate.x, candidate.y)
			d2 = distance(edge[1].x, edge[1].y, candidate.x, candidate.y)
			if (d1 < d2):
				return True, edge[0]
			else:
				return True, edge[1]

	return False, None

def convert_to_graph(trips):
	"""
	Converts a set of trips into a directed graph as defined in graph.py
	"""
	G = Graph()
	for i in range(0, len(trips)):
		t = trips[i]
		prevNode = None
		for j in range(len(t)):
			n = t[j]
			if not isinstance(n, (type, Node)):
				n = Node(n[0], n[1], n[2])
			merge, closest_node = to_merge(n, G, 3, 0.2)
			if merge:
				if prevNode and not G.has_path(prevNode, closest_node, 5):
					G.add_edge(prevNode, closest_node)
				prevNode = closest_node
			else:
				G.add_node(n)
				if prevNode:
					G.add_edge(prevNode, n)
				prevNode = n
	return G