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



def get_training_data(k, location):
	'''
	Get training data from k files from string location.
	E.g. get_training_data(4, "DR_USA_Roundabout_EP")
	Returns a dataframe of k files compiled together +
	A list of all traces
	'''
	frames = []
	traces = []

	for i in range(k):
		path = os.path.join("interaction-dataset-copy/recorded_trackfiles/"
			+location, "vehicle_tracks_00"+str(i)+".csv")
		data = pd.read_csv(path)
		box = [[960, 1015], [980, 1040]]
		data = data.loc[(data['x'] > box[0][0]) & (data['x'] < box[0][1])
    					& (data['y'] > box[1][0]) & (data['y'] < box[1][1])]
		frames.append(data)

		for j in range(100):
			temp = data.loc[(data['track_id'] == j)]
			temp = temp.to_numpy()
			temp = np.vstack((temp[:, 4], temp[:, 5], temp[:, 2], temp[:, 6], temp[:, 7])).T
			traces.append(temp)


		traces_copy = []
		for i in traces:
			if len(i) != 0:
				traces_copy.append(i)


		x_axis = [1, 0]
		for i in range(len(traces_copy)):
			for j in range(len(traces_copy[i])):
				velocity = traces_copy[i][j][3:]
				length = np.linalg.norm(velocity)
				if (length == 0):
					heading = 0
				else:
					heading = np.arccos(np.dot(velocity/length, [1, 0]))
				traces_copy[i][j][3] = heading



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


if __name__ == "__main__":
	(data, traces) = get_training_data(1, "DR_USA_Roundabout_EP")
	






	








