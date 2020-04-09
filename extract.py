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
	plt.show()

def get_clusters(data):
	'''
	Perform k-means clustering on data.
	Returns array of clusters (coordinates).
	'''
	


if __name__ == "__main__":
	data = get_training_data(4, "DR_USA_Roundabout_EP")



