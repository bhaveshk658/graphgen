import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.animation as animation

import collections

from sklearn.cluster import KMeans

from math import hypot


import os
import time

def distance(x1, y1, x2, y2):
	return pow((pow(x1 - x2, 2) + pow(y1 - y2, 2)), 0.5)


def arclength(f, a, b, tol=1e-6):
    """Compute the arc length of function f(x) for a <= x <= b. Stop
    when two consecutive approximations are closer than the value of
    tol.
    """
    nsteps = 1  # number of steps to compute
    oldlength = 1.0e20
    length = 1.0e10
    while abs(oldlength - length) >= tol:
        nsteps *= 2
        fx1 = f(a)
        xdel = (b - a) / nsteps  # space between x-values
        oldlength = length
        length = 0
        for i in range(1, nsteps + 1):
            fx0 = fx1  # previous function value
            fx1 = f(a + i * (b - a) / nsteps)  # new function value
            length += hypot(xdel, fx1 - fx0)  # length of small line segment
    return length



### Basic parameters ###

if __name__ == "__main__":

    data = pd.read_csv(os.path.join("interaction-dataset-copy/recorded_trackfiles/DR_USA_Roundabout_EP", "vehicle_tracks_000.csv"))
    box = [[975, 1000], [985, 1010]]
    box2 = [[1055, 1085], [1005, 1025]]

    ### Plot all data points in the csv ###

    total_data = data.to_numpy()
    total_data = np.vstack((total_data[:, 4], total_data[:, 5])).T
    plt.figure(1)
    #plt.scatter(total_data[:, 0], total_data[:, 1], s=1)
    plt.title("All data points + K-means clusters")
    plt.xlabel("X-coordinates")
    plt.ylabel("Y-coordinates")


    ### K-means clustering ###

    Kmean = KMeans(n_clusters=250)
    Kmean.fit(total_data)
    centroids = Kmean.cluster_centers_
    for i in range(len(centroids)):
    	plt.scatter(centroids[i][0], centroids[i][1], s=10)

    ###Cars of interest - 16, 23 ###

    d16 = data.loc[(data['track_id'] == 16) & (data['x'] > box[0][0]) & (data['x'] < box[0][1])
    	& (data['y'] > box[1][0]) & (data['y'] < box[1][1])].to_numpy()

    d16 = np.vstack((d16[:, 4], d16[:, 5], d16[:, 2])).T

    d23 = data.loc[(data['track_id'] == 23) & (data['x'] > box[0][0]) & (data['x'] < box[0][1])
    	& (data['y'] > box[1][0]) & (data['y'] < box[1][1])].to_numpy()
    d23 = np.vstack((d23[:, 4], d23[:, 5], d23[:, 2])).T


    ### Predict cluster of each data point in box ###
    '''
    Find merge point by iterating through centroids
    Fit curve to car 16's clusters and fit another to car 23's clusters
    	Represent functions for lanes/the road
    Use actual x-coordinates in d16/d23 to determine locations on the road using functions
    	Get arc length from that point to merge point (either is already on function or recalculate)
    	Map timestamp to arc length

    Plot arc length (distance) vs time



    '''
    interest16 = Kmean.predict(d16[:, :-1])
    interest23 = Kmean.predict(d23[:, :-1])

    ### Find merge point and coordinates of the clusters that the cars travel through###

    cluster_coord_16 = []
    cluster_coord_23 = []

    for i in range(len(centroids)):
    	if i in interest16:
    		cluster_coord_16.append(centroids[i])
    	if i in interest23:
    		cluster_coord_23.append(centroids[i])

    cluster_coord_16 = np.array(cluster_coord_16)
    cluster_coord_23 = np.array(cluster_coord_23)

    poly_16 = np.polyfit(cluster_coord_16[:, 0], cluster_coord_16[:, 1], 3)
    poly_23 = np.polyfit(cluster_coord_23[:, 0], cluster_coord_23[:, 1], 4)

    d16_pred = np.vstack((d16[:, 0], np.polyval(poly_16, d16[:, 0]), d16[:, 2])).T
    plt.plot(d16_pred[:, 0], d16_pred[:, 1], c='b')

    d23_pred = np.vstack((d23[:, 0], np.polyval(poly_23, d23[:, 0]))).T
    plt.plot(d23_pred[:, 0], d23_pred[:, 1], c='r')

    temp_x = np.linspace(975, 1100, 200)
    temp_y1 = np.polyval(poly_16, temp_x)
    temp_y2 = np.polyval(poly_23, temp_x)

    idx = np.argwhere(np.diff(np.sign(temp_y2 - temp_y1))).flatten()

    merge_point = [temp_x[idx], temp_y1[idx]]

    plt.scatter(cluster_coord_16[:, 0], cluster_coord_16[:, 1], c='b')
    plt.scatter(cluster_coord_23[:, 0], cluster_coord_23[:, 1], c='r')
    plt.scatter(merge_point[0], merge_point[1], c='g')

    ### Isolate points with proper timestamps ###
    d23_time = []

    for i in range(len(d23)):
    	if d23[i][2] in d16[:, 2]:
    		d23_time.append(d23[i])

    d23 = np.array(d23_time)

    d23_pred = np.vstack((d23[:, 0], np.polyval(poly_23, d23[:, 0]), d23[:, 2])).T


    ### Redefine as functions ###
    def f(x):
    	return np.polyval(poly_16, x)

    def g(x):
    	return np.polyval(poly_23, x)

    dist_16 = []
    dist_23 = []
    ### Compute arc lengths ###
    for i in d16_pred:
    	dist_16.append([i[2], arclength(f, i[0], merge_point[0])])
    for j in d23_pred:
    	dist_23.append([j[2], arclength(g, j[0], merge_point[0])])

    dist_16 = np.array(dist_16)
    dist_23 = np.array(dist_23)



    plt.figure(2)
    plt.title("Distance to merge point versus time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Distance to point on path closest to merge point")
    plt.plot(dist_16[:, 0], dist_16[:, 1], c='b')
    plt.plot(dist_23[:, 0], dist_23[:, 1], c='r')

    plt.show()



