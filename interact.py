import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.animation as animation

import collections

from sklearn.cluster import KMeans


import os
import time

def distance(x1, y1, x2, y2):
	return pow((pow(x1 - x2, 2) + pow(y1 - y2, 2)), 0.5)

### Basic parameters ###

data = pd.read_csv(os.path.join("interaction-dataset-copy/recorded_trackfiles/DR_USA_Roundabout_EP", "vehicle_tracks_000.csv"))
box = [[975, 1000], [985, 1010]]
box2 = [[1055, 1085], [1005, 1025]]

### Plot all data points in the csv ###

total_data = data.to_numpy()
total_data = np.vstack((total_data[:, 4], total_data[:, 5])).T
plt.figure(1)
plt.scatter(total_data[:, 0], total_data[:, 1], s=1)
plt.title("All data points + K-means clusters")
plt.xlabel("X-coordinates")
plt.ylabel("Y-coordinates")


### K-means clustering ###

Kmean = KMeans(n_clusters=200)
Kmean.fit(total_data)
centroids = Kmean.cluster_centers_
for i in range(len(centroids)):
	plt.scatter(centroids[i][0], centroids[i][1], s=20, c='r')

###Cars of interest - 16, 23 ###

d16 = data.loc[(data['track_id'] == 16) & (data['x'] > box[0][0]) & (data['x'] < box[0][1])
	& (data['y'] > box[1][0]) & (data['y'] < box[1][1])].to_numpy()

d16 = np.vstack((d16[:, 4], d16[:, 5], d16[:, 2])).T

d23 = data.loc[(data['track_id'] == 23) & (data['x'] > box[0][0]) & (data['x'] < box[0][1])
	& (data['y'] > box[1][0]) & (data['y'] < box[1][1])].to_numpy()
d23 = np.vstack((d23[:, 4], d23[:, 5], d23[:, 2])).T

### Predict cluster of each data point in box ###

interest16 = Kmean.predict(d16[:, :-1])
interest23 = Kmean.predict(d23[:, :-1])

### Find merge point ###

for i in range(len(centroids)):
	if i in interest23 and i in interest16:
		merge_point = centroids[i]
		break

plt.scatter(merge_point[0], merge_point[1], c='g')

### Isolate points with proper timestamps ###
d23_time = []

for i in range(len(d23)):
	if d23[i][2] in d16[:, 2]:
		d23_time.append(d23[i])

d23 = np.array(d23_time)

### Find point closest to merge point ###

dist_16 = np.array([distance(merge_point[0], merge_point[1], i[0], i[1]) for i in d16])
closest_16 = d16[np.argmin(dist_16)]
dist_23 = np.array([distance(merge_point[0], merge_point[1], i[0], i[1]) for i in d23])
closest_23 = d16[np.argmin(dist_23)]


plt.figure(2)
plt.title("Distance to merge point versus time")
plt.xlabel("Time (ms)")
plt.ylabel("Distance to point on path closest to merge point")
plt.plot(d16[:, 2], dist_16, c='r')
plt.plot(d23[:, 2], dist_23, c='b')

plt.show()



