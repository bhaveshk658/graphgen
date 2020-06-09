import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.animation as animation
from matplotlib import cm

import collections

from sklearn.cluster import KMeans

from math import hypot

from dipy.segment.clustering import QuickBundles


import os
import time

#def direction(a, b, c):


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


def plot_all_data(data):
    '''
    Plots data returned by get_training_data.
    Allows us to get visualization of what the map can look like.
    Isolates x and y coordinates from dataframe and plots.
    '''
    for trace in data:
        plt.scatter(trace[:, 0], trace[:, 1], c='r')
    

def get_clusters(data):
	'''
	Perform k-means clustering on data.
	Returns array of clusters (coordinates).
	'''
	Kmean = KMeans(n_clusters=10)
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


def quick_bundles(data):
    qb = QuickBundles(threshold=10)
    clusters=qb.cluster(data)
    '''
    color=iter(cm.rainbow(np.linspace(0,1,len(clusters))))
    for i in range(len(clusters)):
        c = next(color)
        if len(clusters[i].indices) < 4:
            continue
        for j in clusters[i].indices:
            plt.plot(data[j][:, 0], data[j][:, 1], c=c)
    '''
    return clusters


def edge_heading(p1, p2):
    '''
    Direction of an edge from node p1 to node p2.
    '''
    vector = np.array([p2.x-p1.x, p2.y-p1.y])
    length = np.linalg.norm(vector)
    if (length == 0):
        return 0
    else:
        return np.arccos(np.dot(vector/length, [1, 0]))


def edge_dist(p1, p2, p3):
    '''
    Distance from point p3 to line segment p1-p2.
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    dist = np.linalg.norm(p2 - p1)
    if dist == 0:
        return distance(p1[0], p1[1], p3[0], p3[1])
    return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)

def node_dist(node1, node2):
    '''
    Distance between two nodes.
    '''
    return distance(node1.x, node1.y, node2.x, node2.y)

def dist(point1, point2):
    '''
    Distance between two points represented by arrays.
    '''
    return distance(point1[0], point1[1], point2[0], point2[1])


def dist_point_to_line(candidate, node1, node2):
    '''
    Distance from a candidate node to an edge.
    '''

    area = abs(0.5*np.linalg.det(np.array([[candidate.x, candidate.y, 1],
                                       [node1.x, node1.y, 1],
                                       [node2.x, node2.y, 1]])))
    distance = ((node2.y - node1.y)**2 + (node2.x - node1.x)**2)**0.5

    return 2*area/distance

def line_line_segment_intersect(p, d, p1, p2):
    '''
    p: point on the line.
    d: direction vector of line of interest.
    p1: first endpoint of line segment.
    p2: second endpoint of line segment.
    '''
    p = np.array(p)
    p1 = np.array(p1)
    p2 = np.array(p2)
    d = np.array(d)

    v1 = np.dot(d, p1-p)
    v2 = np.dot(d, p2-p)

    if (v1 >= 0 and v2 >= 0) or (v1 < 0 and v2 < 0):
        return True
    else:
        return False


