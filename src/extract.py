import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *

import os

import utils
from utils import dist
from utils import dist_point_to_line

from node import Node

import networkx as nx

from sklearn.neighbors import KDTree

def get_training_data(n, location):
	'''
	Get training data from n files from string location.
	E.g. get_training_data(4, "DR_USA_Roundabout_EP")
	Returns a dataframe of n files compiled together +
	A list of all traces
	'''
	frames = []
	traces = []

	for i in range(n):
		path = os.path.join("interaction-dataset-copy/recorded_trackfiles/"
			+location, "vehicle_tracks_00"+str(i)+".csv")
		data = pd.read_csv(path)

		# Define rectangle of area to take points from.
		box = [[960, 1015], [980, 1040]]
		data = data.loc[(data['x'] > box[0][0]) & (data['x'] < box[0][1]) & (data['y'] > box[1][0]) & (data['y'] < box[1][1])]
		frames.append(data)

		# Add the trace for each car j to the list of traces.
		# Contains x, y, x-velocity, y-velocity.
		for j in range(len(data.index)):
			temp = data.loc[(data['track_id'] == j)]
			temp = temp.to_numpy()
			if len(temp != 0):
				temp = np.vstack((temp[:, 4], temp[:, 5], temp[:, 6], temp[:, 7])).T
				traces.append(temp)

		# Get headings using velocity vector at each point.
		for i in range(len(traces)):
			for j in range(len(traces[i])):
				velocity = traces[i][j][2:]
				length = np.linalg.norm(velocity)
				if (length == 0):
					heading = 0
				else:
					x = velocity/length
					heading = np.arccos(np.dot(x, [1, 0]))
				traces[i][j][3] = heading
		
		# Keep traces as x, y, heading.
		for i in range(len(traces)):
			traces[i] = np.delete(traces[i], 2, axis=1)

	return (pd.concat(frames), traces)

def to_merge(candidate, G):
	'''
	Determines if a candidate node should be merged into the graph.
	Returns whether or not to merge, the target edge,
	the closest node, and the distance.
	'''
	if len(G.edges) == 0:
		return (False, 0, 0, 0)
	edges = [e for e in G.edges]

	# Find edges that satisfy merge conditions.
	for edge in edges:
		temp_dist = dist_point_to_line(edge[0], edge[1], candidate)
		temp_heading = abs(candidate.heading - edge[0].heading)
		
		# Check merge parameters.
		if temp_dist < 3 and temp_heading < 0.4:
			d1 = utils.distance(edge[0].x, edge[0].y, candidate.x, candidate.y)
			d2 = utils.distance(edge[1].x, edge[1].y, candidate.x, candidate.y)
			if (d1 < d2):
				return (True, edge, edge[0], d1)
			else:
				return (True, edge, edge[1], d2)

	return (False, edge, edge[0], 0)

def type_1_force(t):
	'''
	Calculates the type 1 force as described by Cao and Krumm
	for a distance t.
	'''
	M = 10
	N = 20
	s1 = 5
	s2 = 5
	sig_square = s1**2 + s2**2

	return ((M*N)/(sqrt(2*pi*sig_square))) * ((2*t*exp((-t**2)/(2*sig_square)))/(2*sig_square))

def type_2_force(t, orig):
	'''
	Calculates the type 2 force as described by Cao and Krumm.
	'''
	return 0.5*utils.distance(t[0], t[1], orig[0], orig[1])

if __name__ == "__main__":
	(data, traces) = get_training_data(1, "DR_USA_Roundabout_EP")
	
	# Preprocessing: eliminate traces with less than 50 points
	# and thin out traces.
	trips = []
	for c in traces:
		if (len(c) < 50):
			continue
		trip = []
		point = c[0]
		for i in range(1, len(c)):
			if dist(point, c[i]) < 3:
				continue
			trip.append(np.array((c[i][0], c[i][1], c[i][2])))
			point = c[i]
		trips.append(np.array(trip))
	trips = np.array(trips)
	
	# k-d tree requires flattened array.
	lengths = [len(trip) for trip in trips]
	points = np.array([item for sublist in trips for item in sublist])
	headings = points[:, 2]
	points = points[:, :2]

	# Plot points.
	plt.figure(1)
	plt.title('Points before clarification')
	for point in points:
		plt.scatter(point[0], point[1], c='b')
	

	tree = KDTree(points, leaf_size=2)
	original = np.copy(points)

	# Cao and Krumm's preprocessing based on gravitational attraction.
	count = 50
	while count > 0:
		for i in range(1, len(points) - 1):
			a = points[i]
			orig = original[i]
			prev_point = points[i - 1]
			next_point = points[i + 1]
			heading = next_point - prev_point
			d = np.array((heading[1], -heading[0]))

			ind = tree.query_radius(np.array([a]), r=3)

			# Find consecutive points in the list of indices within radius of a.
			pairs = []
			for j in range(len(ind[0])):
				for k in range(j + 1, len(ind[0])):
					if ind[0][j]+1 == ind[0][k]:
						pairs.append([ind[0][j], ind[0][k]])
			
			# Calculate t1 and t2 forces for each pair.
			for pair in pairs:
				b = points[pair[0]]
				c = points[pair[1]]
				seg_heading = c - b
				midpoint = (c-b)/2

				if utils.line_line_segment_intersect(a, d, b, c):
					t1_distance = utils.minDistance(b, c, a)
					t1_direction = (midpoint-a)/(utils.distance(midpoint[0], midpoint[1], a[0], a[1]))
					t1 = np.array(type_1_force(t1_distance) * t1_direction)

					if all(a == orig):
						t2 = 0
					else:
						t2_direction = utils.direction(a, orig)
						t2 = type_2_force(a, orig) * t2_direction

					resultant = t1 + t2
					
					# Limit repelling force to only affect left side of trace.
					angle = np.dot(seg_heading, heading)/(np.linalg.norm(seg_heading)*np.linalg.norm(heading))
					if cos(angle) < 0 and not utils.is_left(b, c, a):
						resultant = 0
					else:
						resultant = resultant*cos(angle)

					a += (5*resultant)
				
			points[i] = a

		count -= 1

	# Convert flattened modified points to traces of Nodes.
	trips = []
	for l in lengths:
		trip = []
		temp = points[:l]
		temp_headings = headings[:l]
		for i in range(len(temp)):
			trip.append(Node(temp[i][0], temp[i][1], temp_headings[i]))
		trips.append(np.array(trip))
		points = points[l:]
		headings = headings[l:]
	
	# Cao and Krumm graph generation algorithm.
	count = 0
	G = nx.MultiDiGraph()
	for i in range(0, len(trips)):
		t = trips[i]
		prevNode = None
		for n in t:
			(merge, closest_edge, closest_node, short_projection_distance) = to_merge(n, G)
			if merge:
				count += 1
				if prevNode is not None and ((not nx.has_path(G, prevNode, closest_node)) or len(nx.shortest_path(G, prevNode, closest_node))) > 5:
					G.add_edge(prevNode, closest_node, weight=1)
					prevNode = closest_node
			else:
				G.add_node(n)
				if prevNode is not None:
					G.add_edge(prevNode, n, weight=1)
				prevNode = n

	# Visualize graph.
	pos = {}
	for node in G:
		pos[node] = [node.x, node.y]
	plt.figure(3)
	plt.title('Generated Graph')
	nx.draw(G, pos, node_size=10)
		
	plt.show()



	
	















	


	






	








