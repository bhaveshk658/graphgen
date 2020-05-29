import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

import os

from utils import distance
from utils import dist
from utils import dist_point_to_line
from utils import quick_bundles
from utils import get_clusters
from utils import plot_clusters
from utils import plot_all_data

from dipy.segment.clustering import QuickBundles

import networkx as nx

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
		box = [[960, 1015], [980, 1040]]
		#box = [[0, 10000], [0, 10000]]
		data = data.loc[(data['x'] > box[0][0]) & (data['x'] < box[0][1]) & (data['y'] > box[1][0]) & (data['y'] < box[1][1])]
		frames.append(data)

		'''
		for j in range(1000):
			temp = data.loc[(data['track_id'] == j)]
			temp = temp.to_numpy()
			temp = np.vstack((temp[:, 4], temp[:, 5], temp[:, 2], temp[:, 6], temp[:, 7])).T
			traces.append(temp)


		traces_copy = []
		for trace in traces:
			if len(trace) != 0:
				traces_copy.append(trace)

		x_axis = [1, 0]
		for j in range(len(traces_copy)):
			for k in range(len(traces_copy[j])):
				velocity = traces_copy[j][k][3:]
				length = np.linalg.norm(velocity)
				if (length == 0):
					heading = 0
				else:
					heading = np.arccos(np.dot(velocity/length, [1, 0]))
				traces_copy[j][k][3] = heading
		'''
		for j in range(len(data.index)):
			temp = data.loc[(data['track_id'] == j)]
			temp = temp.to_numpy()
			if len(temp != 0):
				temp = np.vstack((temp[:, 4], temp[:, 5], temp[:, 6], temp[:, 7])).T
				traces.append(temp)

		for i in range(len(traces)):
			for j in range(len(traces[i])):
				velocity = traces[i][j][2:]
				length = np.linalg.norm(velocity)
				if (length == 0):
					heading = 0
				else:
					heading = np.arccos(np.dot(velocity/length, [1, 0]))
				traces[i][j][3] = heading
		for i in range(len(traces)):
			traces[i] = np.delete(traces[i], 2, axis=1)

		
		
				
	return (pd.concat(frames), traces)

class Node:

	def __init__(self, x, y, heading):
		self.x = x
		self.y = y
		self.heading = heading
		self.merged = [self]

	def merge(self, node):
		self.merged.append(node)
		prev_x = [i.x for i in self.merged]
		prev_y = [i.y for i in self.merged]
		prev_heading = [i.heading for i in self.merged]
		self.x = sum(prev_x)/len(prev_x)
		self.y = sum(prev_y)/len(prev_y)
		self.heading = sum(prev_heading)/len(prev_heading)


def to_merge(candidate, G):
	if len(G.edges) == 0:
		return (False, 0, 0, 0)
	edges = [e for e in G.edges]
	closest = edges[0]
	closest_dist = dist_point_to_line(candidate, closest[0], closest[1])
	for edge in edges[1:]:
		temp_dist = dist_point_to_line(candidate, edge[0], edge[1])
		if temp_dist < closest_dist:
			closest = edge
	edge_vector = np.array([closest[1].x - closest[0].x, closest[1].y - closest[0].y])
	candidate_vector = np.array([candidate.x - closest[0].x, candidate.y - closest[0].y])

	proj = np.multiply((np.dot(candidate_vector, edge_vector)/np.dot(edge_vector, edge_vector)), edge_vector)
	proj_point = np.add(proj, np.array([closest[0].x, closest[0].y]))

	h = dist_point_to_line(candidate, closest[0], closest[1])
	d1 = distance(closest[0].x, closest[0].y, proj_point[0], proj_point[1])
	d2 = distance(closest[1].x, closest[1].y, proj_point[0], proj_point[1])
	angle = abs(candidate.heading - closest[0].heading)

	if (h < 0.2) and (angle < 0.4): #decrease h?
		if (d1 < d2):
			return (True, closest, closest[0], d1)
		else:
			return (True, closest, closest[0], d2)

	return (False, closest, closest[0], d1)







if __name__ == "__main__":
	(data, traces) = get_training_data(1, "DR_USA_Roundabout_EP")
	#points = np.array([item for sublist in traces for item in sublist])
	#points = np.delete(points, [2, 4], axis=1)
	G = nx.DiGraph()
	
	#Preprocessing
	trips = []
	for c in traces:
		if (len(c) < 50):
			continue
		trip = []
		point = c[0]
		for i in range(1, len(c)):
			if dist(point, c[i]) < 1:
				continue
			trip.append(Node(c[i][0], c[i][1], c[i][-1]))
			point = c[i]

		trips.append(np.array(trip))
	
	
	#Algorithm based on paper by Cao and Krumm, Microsoft
	for t in trips:
		prevNode = None
		for n in t:
			(merge, closest_edge, closest_node, short_projection_distance) = to_merge(n, G)
			if merge:
				if prevNode is not None and nx.has_path(G, prevNode, closest_node) and len(nx.shortest_path(G, prevNode, closest_node)) > 5:
					G.add_edge(prevNode, closest_node)
					prevNode = closest_node
			else:
				G.add_node(n)
				if prevNode is not None:
					G.add_edge(prevNode, n)
				prevNode = n

	pos = {}
	for node in G:
		pos[node] = [node.x, node.y]
	
	nx.draw(G, pos, node_size=10)
	

	#plt.show()


















	


	






	








