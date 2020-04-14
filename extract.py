import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from math import hypot

import os
import time

from interact_toolset import distance
from frechetdist import frdist

from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric

  box = [[960, 1015], [980, 1040]]

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
		frames.append(data)

		for j in range(len(data.index)):
			temp = data.loc[(data['track_id'] == j)]
			temp = temp.to_numpy()
			temp = np.vstack((temp[:, 4], temp[:, 5])).T
			traces.append(temp)



	return (pd.concat(frames), traces)

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
	temp = data.to_numpy()
	temp = np.vstack((temp[:, 4], temp[:, 5])).T
	Kmean = KMeans(n_clusters=250)
	Kmean.fit(temp)
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




if __name__ == "__main__":
	(data, traces) = get_training_data(1, "DR_USA_Roundabout_EP")
	temp = []
	for i in traces:
		if len(i) != 0:
			temp.append(i)
	feature = ResampleFeature(nb_points=256)
	metric = FrechetDistance()
	qb = QuickBundles(threshold=7, metric=metric)
	clusters = qb.cluster(temp)
	








