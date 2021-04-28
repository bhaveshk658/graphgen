from graphgen.generate.utils import distance, dist_point_to_line, dist_point_to_line_nx
from graphgen.graph.graph import Graph
from graphgen.graph.node import Node

import networkx as nx

def to_merge_nx(candidate, G, dist_limit, heading_limit):
	"""
	Determines if a candidate node should be merged into the graph.
	Returns whether or not to merge, the target edge,
	the closest node, and the distance.
	"""
	if len(G.edges) == 0:
		return None
	
	edges = G.edges

	for edge in edges:
		temp_dist = dist_point_to_line_nx(edge[0], edge[1], candidate)
		temp_heading = abs(candidate['heading'] - edge[0]['heading'])

		if temp_dist < dist_limit and temp_heading < heading_limit:
			d1 = distance(edge[0]['x'], edge[0]['y'], candidate['x'], candidate['y'])
			d2 = distance(edge[1]['x'], edge[1]['y'], candidate['x'], candidate['y'])
			if d1 < d2:
				return edge[0]
			else:
				return edge[1]

		return None

def convert_to_graph_nx(trips, dist_limit=3, heading_limit=0.78):
	"""
	Converts a set of trips into a directed graph.
	"""
	G = nx.DiGraph()
	for t in trips:
		prevNode = None
		for n in t:
			closest_node = to_merge_nx(n, G, dist_limit, heading_limit)
			if closest_node:
				# Update closest node
				if prevNode:
					if not nx.has_path(G, prevNode, closest_node) or \
						nx.shortest_path_length(G, prevNode, closest_node) < 5:
						G.add_edge(prevNode, closest_node)
				prevNode = closest_node
			else:
				G.add_node(n)
				if prevNode:
					G.add_edge(prevNode, n)
				prevNode = n
	return G




def to_merge(candidate, G, dist_limit, heading_limit):
	"""
	Determines if a candidate node should be merged into the graph.
	Returns whether or not to merge, the target edge,
	the closest node, and the distance.
	"""
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

def convert_to_graph(trips, dist_limit=3, heading_limit=0.78):
	"""
	Converts a set of trips into a directed graph.
	"""
	G = Graph()
	for t in trips:
		prevNode = None
		for n in t:
			merge, closest_node = to_merge(n, G, dist_limit, heading_limit)
			if merge:
				closest_node.update(n)
				if prevNode and not G.has_path(prevNode, closest_node, 5):
					G.add_edge(prevNode, closest_node)
				prevNode = closest_node
			else:
				G.add_node(n)
				if prevNode:
					G.add_edge(prevNode, n)
				prevNode = n
	return G
