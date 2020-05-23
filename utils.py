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

def compute_headings(data):
    headings = []
    velocities = data[:, 3:]

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


def plot_clusters(data):
    '''
    Perform k-means clustering on data and plots.
    No return.
    '''
    centroids = get_clusters(data)
    for i in range(len(centroids)):
        plt.scatter(centroids[i][0], centroids[i][1], s=10, c='r')


def quick_bundles(data): #### need to modify for updated data ####
    qb = QuickBundles(threshold=10)
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


def edge_heading(p1, p2):
    vector = np.subtract(p2, p1)
    length = np.linalg.norm(vector)
    if (length == 0):
        return 0
    else:
        return np.arccos(np.dot(vector/length, [1, 0]))


def edge_dist(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return distance(p1[0], p1[1], p3[0], p3[1])
    return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)

def node_dist(node1, node2):

    return distance(node1.x, node1.y, node2.x, node2.y)

def dist(point1, point2):

    return distance(point1[0], point1[1], point2[0], point2[1])


def dist_point_to_line(candidate, node1, node2):
    area = abs(0.5*np.linalg.det(np.array([[candidate.x, candidate.y, 1],
                                       [node1.x, node1.y, 1],
                                       [node2.x, node2.y, 1]])))
    distance = ((node2.y - node1.y)**2 + (node2.x - node1.x)**2)**0.5

    return 2*area/distance


