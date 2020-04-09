import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from math import hypot

import os
import time

from interact_toolset import distance

def get_training_data(k, location):
	'''
	Get training data from k files from string location.
	E.g. get_training_data(4, "DR_USA_Roundabout_EP")
	Returns a dataframe of k files compiled together.
	'''
	frames = []

	for i in range(k):
		data = pd.read_csv(os.path.join("interaction-dataset-copy/recorded_trackfiles/"
			+location, "vehicle_tracks_00"+str(k)+".csv"))
		frames.append(data)

	return pd.concat(frames)

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
	data = get_training_data(2, "DR_USA_Roundabout_EP")
	plot_all_data(data)
	plot_clusters(data)
	plt.show()




