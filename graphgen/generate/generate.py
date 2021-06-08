from graphgen.generate.utils import distance, dist_point_to_line, dist_point_to_line_nx
from graphgen.graph.graph import Graph
from graphgen.graph.node import Node

import networkx as nx

def to_merge_nx(candidate, G, dist_limit, heading_limit):
	"""
	candidate: [x, y]
	G: nx.DiGraph()
	dist_limit: number
	heading_limit: number
	"""
	if len(G.edges) == 0:
		return None

	edges = G.edges

	for e in edges:
		temp_dist = dist_point_to_line_nx(G, e[0], e[1], candidate)
		temp_heading = abs(candidate[2] - G.nodes[e[0]]['heading'])

		if temp_dist < dist_limit and temp_heading < heading_limit:
			d1 = distance(G.nodes[e[0]]['x'], G.nodes[e[0]]['y'], candidate[0], candidate[1])
			d2 = distance(G.nodes[e[1]]['x'], G.nodes[e[1]]['y'], candidate[0], candidate[1])

			if d1 < d2:
				return e[0]
			else:
				return e[1]
	return None

def convert_to_graph_nx(trips, dist_limit=3, heading_limit=0.78):
	"""
	Converts a set of trips into a directed graph.
	"""
	G = nx.DiGraph()
	node_count = 0
	for t in trips:
		prevNode = None
		for n in t:
			closest_node = to_merge_nx(n, G, dist_limit, heading_limit)
			if closest_node:
				if prevNode:
					path_exists = nx.has_path(G, prevNode, closest_node)
					path_is_short = path_exists and nx.dijkstra_path_length(G, prevNode, closest_node) < 5

					if not path_exists or not path_is_short:
						G.add_edge(prevNode, closest_node, volume=1)
					elif path_is_short:
						path = nx.dijkstra_path(G, prevNode, closest_node)
						for i in range(len(path)-1):
							G.edges[path[i], path[i+1]]['volume'] += 1
				prevNode = closest_node
			else:
				G.add_node(node_count, x=n[0], y=n[1], heading=n[2], pos = (n[0], n[1]))
				if prevNode:
					G.add_edge(prevNode, node_count, volume=1)
				prevNode = node_count
				node_count += 1
	return G

def clean(G, volume_threshold):
	"""
	Removes all edges in G with volume < volume_threshold.
	"""
	H = G.copy()
	edges_to_remove = [e for e in H.edges if H.edges[e]['volume'] < volume_threshold]
	H.remove_edges_from(edges_to_remove)
	return H




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
