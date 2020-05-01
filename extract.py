import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from math import hypot
from matplotlib.pyplot import cm

import os
import time

from interact_toolset import distance
from frechetdist import frdist

from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric

import random
import networkx as nx


class FrechetDistance(Metric):
	'''
	Computes Frechet Distance between two trajectories.
	'''
	def __init__(self):
		super(FrechetDistance, self).__init__(feature=ResampleFeature(nb_points=256))

	def are_compatible(self, shape1, shape2):
		return len(shape1) == len(shape2)

	def dist(self, v1, v2):
		return frdist(v1, v2)

def compute_headings(data):
	headings = []
	velocities = data[:, 3:]



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
		data = data.loc[(data['x'] > box[0][0]) & (data['x'] < box[0][1]) & (data['y'] > box[1][0]) & (data['y'] < box[1][1])]
		frames.append(data)

		for j in range(100):
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



	return (pd.concat(frames), traces_copy)

def plot_all_data(data):
	'''
	Plots data returned by get_training_data.
	Allows us to get visualization of what the map can look like.
	Isolates x and y coordinates from dataframe and plots.
	'''
	temp = data.to_numpy()
	temp = np.vstack((temp[:, 4], temp[:, 5])).T
	plt.scatter(temp[:, 0], temp[:, 1], s=1)
	plt.title("All Data Points")
	plt.xlabel("X-Coordinate")
	plt.ylabel("Y-Coordinate")

def get_clusters(data):
	'''
	Perform k-means clustering on data.
	Returns array of clusters (coordinates).
	'''
	Kmean = KMeans(n_clusters=15)
	Kmean.fit(data)
	centroids = Kmean.cluster_centers_
	return centroids

def plot_clusters(data):
	'''
	Perform k-means clustering on data and plots.
	No return.
	'''
	centroids = get_clusters(data)
	for i in range(len(centroids)):
		plt.scatter(centroids[i][0], centroids[i][1], s=10, c='r')


def quick_bundles(data): #### need to modify for updated data ####
	qb = QuickBundles(threshold=5)
	clusters = qb.cluster(data)
	color = iter(cm.rainbow(np.linspace(0,1,len(clusters))))
	plt.figure(1)
	for i in range(len(clusters)):
		c = next(color)
		if len(clusters[i].indices) < 4:
			continue
		for j in clusters[i].indices:
			plt.plot(data[j][:, 0], data[j][:, 1], c=c)


	plt.figure(2)
	color = iter(cm.rainbow(np.linspace(0,1,len(clusters))))
	for i in clusters:
		c = next(color)
		lane_traces = data[i.indices[0]]
		for j in i.indices[1:]:
			lane_traces = np.concatenate((lane_traces, data[j]))

		centroids = get_clusters(lane_traces)
		plt.scatter(centroids[:, 0], centroids[:, 1], s=10, c=c)
	
	plt.show()


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



def dist(point1, point2):

	return distance(point1[0], point1[1], point2[0], point2[1])

def node_dist(node1, node2):

	return distance(node1.x, node1.y, node2.x, node2.y)

def edge_dist(p1, p2, p3):
	p1 = np.array(p1)
	p2 = np.array(p2)
	p3 = np.array(p3)
	dist = np.linalg.norm(p2 - p1)
	if dist == 0:
		return distance(p1[0], p1[1], p3[0], p3[1])
	return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)

def edge_heading(p1, p2):
	vector = np.subtract(p2, p1)
	length = np.linalg.norm(vector)
	if (length == 0):
		return 0
	else:
		return np.arccos(np.dot(vector/length, [1, 0]))


def to_merge(candidate, G):
	if len(G.edges) == 0:
		return (False, 0, 0, 0)
	edges = [e for e in G.edges]
	closest = edges[0]
	closest_dist = node_dist(candidate, closest[0]) + node_dist(candidate, closest[1])
	for edge in edges[1:]:
		temp_dist = node_dist(candidate, edge[0]) + node_dist(candidate, edge[1])
		if temp_dist < closest_dist:
			closest = edge
	edge_vector = np.array([closest[1].x - closest[0].x, closest[1].y - closest[0].y])
	candidate_vector = np.array([candidate.x - closest[0].x, candidate.y - closest[0].y])

	proj = np.multiply((np.dot(candidate_vector, edge_vector)/np.dot(edge_vector, edge_vector)), edge_vector)
	proj_point = np.add(proj, np.array([closest[0].x, closest[0].y]))

	h = distance(candidate.x, candidate.y, proj_point[0], proj_point[1])
	d1 = distance(closest[0].x, closest[0].y, proj_point[0], proj_point[1])
	d2 = distance(closest[1].x, closest[1].y, proj_point[0], proj_point[1])
	angle = abs(candidate.heading - closest[0].heading)

	if (h < 50) and (angle < 0.5):
		if (d1 < d2):
			return (True, closest, closest[0], d1)
		else:
			return (True, closest, closest[0], d2)

	return (False, closest, closest[0], d1)






if __name__ == "__main__":
	(data, traces) = get_training_data(4, "DR_USA_Roundabout_EP")
	points = [item for sublist in traces for item in sublist]
	G = nx.DiGraph()
	dist_tol = 30
	head_tol = 0.5

	#Preprocessing
	trips = []
	for c in traces:
		if (len(c) < 50):
			continue
		trip = []
		point = c[0]
		for i in range(1, len(c)):
			if dist(point, c[i]) < 3:
				continue
			trip.append(Node(c[i][0], c[i][1], c[i][-1]))
			point = c[i]

		trips.append(np.array(trip))


	for trip in trips:
		prevNode = None
		for n in trip:
			(merge, closest_edge, closest_node, short_projection_distance) = to_merge(n, G)
			if merge:
				if short_projection_distance > 100:
					G.remove_edge(closest_edge[0], closest_edge[1])
					G.add_edge(closest_node, n)
					G.add_edge(n, closest_node)
					check = True
				else:
					closest_node.merge(n)
					n = closest_node
					check = False
				if check and prevNode is not None and nx.has_path(G, prevNode, n) and len(nx.shortest_path(G, prevNode, n)) > 6:
					G.add_edge(prevNode, n)
					prevNode = n

			else:
				G.add_node(n)
				if prevNode is not None:
					G.add_edge(prevNode, n)
				prevNode = n

	pos = {}
	for node in G:
		pos[node] = [node.x, node.y]

	nx.draw(G, pos, node_size=50)
	plt.show()

















	


	






	








