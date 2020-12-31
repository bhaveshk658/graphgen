from graphgen.generate.utils import distance, dist_point_to_line

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