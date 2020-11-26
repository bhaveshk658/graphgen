import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *
from copy import deepcopy

import os

import utils
from utils import dist, dist_point_to_line, edge_heading, curvature

from node import Node

import networkx as nx

from sklearn.neighbors import KDTree

from graph import Graph

def get_training_data(n, location):
	'''
	Get training data from n files from string location.
	E.g. get_training_data(4, "DR_USA_Roundabout_EP")
	Returns a dataframe of n files compiled together +
	A list of all traces
	'''
	traces = []

	for i in range(n):
		path = os.path.join("interaction-dataset-copy/recorded_trackfiles/"
			+location, "vehicle_tracks_00"+str(i)+".csv")
		data = pd.read_csv(path)

		# Define rectangle of area to take points from.
		box = [[960, 1015], [980, 1040]]
		data = data.loc[(data['x'] > box[0][0]) & (data['x'] < box[0][1]) & (data['y'] > box[1][0]) & (data['y'] < box[1][1])]

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

	return traces

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
			d1 = utils.distance(edge[0].x, edge[0].y, candidate.x, candidate.y)
			d2 = utils.distance(edge[1].x, edge[1].y, candidate.x, candidate.y)
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

def clean(traces, length_threshold, dist_threshold):
	"""
	Clean a list of traces, Eliminate traces with length
	less than length_threshold and eliminate points that are
	within dist_threshold of each other.
	"""
	trips = []
	for c in traces:
		# If there are less than length_threshold points, skip this trace.
		if (len(c) < length_threshold):
			continue
		trip = []
		point = c[0]
		for i in range(1, len(c)):
			# If the point is less than dist_threshold unit away, skip it.
			if dist(point, c[i]) < dist_threshold:
				continue
			trip.append([c[i][0], c[i][1], c[i][2]])
			point = c[i]
		trips.append(trip)

	return trips

if __name__ == "__main__":
	traces = get_training_data(1, "DR_USA_Roundabout_EP")

	# Deepcopy original data to plot if needed.
	data = deepcopy(traces)

	# Preprocessing: eliminate traces with less than 50 points
	# and thin out traces.
	trips = clean(traces, 50, 1)

	# Convert each point to a node.
	for trip in trips:
		for i in range(len(trip)):
			point = trip[i]
			node = Node(point[0], point[1], point[2])
			trip[i] = node
	
	# Plot nodes section by section (done manually).
	plt.xlim(969, 1015)
	plt.ylim(980, 1040)

	rb = [0, 1, 8, 11, 14, 15, 18, 19, 21, 24]
	rb_trips = [trips[i] for i in rb]
	rb_graph = convert_to_graph(rb_trips)
	rb_graph.draw()

	br = [6, 12, 13, 20, 22, 25, 26, 27]
	br_trips = [trips[i] for i in br]
	br_graph = convert_to_graph(br_trips)
	br_graph.draw()

	tr = [2, 7, 10]
	tr_trips = [trips[i] for i in tr]
	tr_graph = convert_to_graph(tr_trips)
	tr_graph.draw()

	rt = [3, 17]
	rt_trips = [trips[i] for i in rt]
	rt_graph = convert_to_graph(rt_trips)
	rt_graph.draw()

	special = [4]
	special_trips = [trips[i] for i in special]
	special_graph = convert_to_graph(special_trips)
	special_graph.draw()

	bt = [9, 23]
	bt_trips = [trips[i] for i in bt]
	bt_graph = convert_to_graph(bt_trips)
	bt_graph.draw()

	plt.show()
				


	



	
	














