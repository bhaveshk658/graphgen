import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import *
from copy import deepcopy

import os

import utils
from utils import dist
from utils import dist_point_to_line

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

	return traces

def to_merge(candidate, G):
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
		if temp_dist < 3 and temp_heading < 0.2:
			d1 = utils.distance(edge[0].x, edge[0].y, candidate.x, candidate.y)
			d2 = utils.distance(edge[1].x, edge[1].y, candidate.x, candidate.y)
			if (d1 < d2):
				return True, edge[0]
			else:
				return True, edge[1]

	return False, None

if __name__ == "__main__":
	traces = get_training_data(1, "DR_USA_Roundabout_EP")
	data = deepcopy(traces)
	
	# Preprocessing: eliminate traces with less than 50 points
	# and thin out traces.
	trips = []
	for c in traces:

		if (len(c) < 50):
			continue

		trip = []
		point = c[0]
		for i in range(1, len(c)):

			if dist(point, c[i]) < 1:
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


	G = Graph()
	for i in range(0, len(trips)):
		t = trips[i]
		prevNode = None
		for n in t:
			merge, closest_node = to_merge(n, G)
			if merge:
				if prevNode and not G.has_path(prevNode, closest_node, 5):
					G.add_edge(prevNode, closest_node)
				prevNode = closest_node
			else:
				G.add_node(n)
				if prevNode:
					G.add_edge(prevNode, n)
				prevNode = n
	
	G_copy = deepcopy(G)

	print("Graphing...")
	
	plt.figure(1)
	plt.title("Raw graph")
	G.draw()
	for trace in data:
		for point in trace:
			plt.scatter(point[0], point[1], c='b', alpha=0.05)
	'''
	plt.figure(2)
	plt.title("Cleanup: Deleting edges based on node heading vs edge heading")
	G.cleanup()
	G.draw()
	for trace in data:
		for point in trace:
			plt.scatter(point[0], point[1], c='b', alpha=0.05)
	'''
	plt.figure(3)
	plt.title("Cleanup: Second order")
	G_copy.second_order_cleanup()
	G_copy.draw()
	
	for trace in data:
		for point in trace:
			plt.scatter(point[0], point[1], c='b', alpha=0.05)
	
	plt.show()
				


	



	
	














