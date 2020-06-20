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
from math import sqrt

from dipy.segment.clustering import QuickBundles


import os
import time

def distance(x1, y1, x2, y2):
    '''
    Distance between two points (x1, y1), (x2, y2).
    '''
    return pow((pow(x1 - x2, 2) + pow(y1 - y2, 2)), 0.5)

def dist(p1, p2):
    '''
    Distance between two points represented by arrays.
    '''
    return distance(p1[0], p1[1], p2[0], p2[1])

def direction(p1, p2):
    '''
    Direction between two points p1 and p2.
    '''
    p1 = np.array(p1)
    p2 = np.array(p2)
    return (p2-p1)/distance(p1[0], p1[1], p2[0], p2[1])

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

def dist_point_to_line(A, B, E):
    '''
    Distance from Node E to edge A->B. Adapted from GeeksforGeeks.
    '''
    A = [A.x, A.y]
    B = [B.x, B.y]
    E = [E.x, E.y]
    # vector AB  
    AB = [None, None];  
    AB[0] = B[0] - A[0];  
    AB[1] = B[1] - A[1];  
  
    # vector BP  
    BE = [None, None]; 
    BE[0] = E[0] - B[0];  
    BE[1] = E[1] - B[1];  
  
    # vector AP  
    AE = [None, None]; 
    AE[0] = E[0] - A[0]; 
    AE[1] = E[1] - A[1];  
  
    # Variables to store dot product  
  
    # Calculating the dot product  
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1];  
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1];  
  
    # Minimum distance from  
    # point E to the line segment  
    reqAns = 0;  
  
    # Case 1  
    if (AB_BE > 0) : 
  
        # Finding the magnitude  
        y = E[1] - B[1];  
        x = E[0] - B[0];  
        reqAns = sqrt(x * x + y * y);  
  
    # Case 2  
    elif (AB_AE < 0) : 
        y = E[1] - A[1];  
        x = E[0] - A[0];  
        reqAns = sqrt(x * x + y * y);  
  
    # Case 3  
    else: 
  
        # Finding the perpendicular distance  
        x1 = AB[0];  
        y1 = AB[1];  
        x2 = AE[0];  
        y2 = AE[1];  
        mod = sqrt(x1 * x1 + y1 * y1);  
        reqAns = abs(x1 * y2 - y1 * x2) / mod;  
      
    return reqAns

def ccw(A, B, C):
    '''
    Check if a point C is counter-clockwise to AB.
    '''
    return (C[1] - A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def is_intersect(A, B, C, D):
    '''
    Check if two line segments AB and CD intersect.
    '''
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def line_line_segment_intersect(p, d, p1, p2):
    '''
    Check if a line defined by point p and direction d intersects
    a segment p1p2.
    '''
    p = np.array(p)
    p1 = np.array(p1)
    p2 = np.array(p2)
    d = np.array(d)

    extend1 = p + 500*d
    extend2 = p - 500*d

    return is_intersect(p, extend1, p1, p2) or is_intersect(p, extend2, p1, p2)


def minDistance(A, B, E) :  
    '''
    Minimum distance from a point E to line segment AB. Adapted from GeeksforGeeks.
    '''
    A = np.array(A)
    B = np.array(B)
    E = np.array(E)
  
    # vector AB  
    AB = [None, None]   
    AB[0] = B[0] - A[0]   
    AB[1] = B[1] - A[1]   
  
    # vector BP  
    BE = [None, None]  
    BE[0] = E[0] - B[0]   
    BE[1] = E[1] - B[1]   
  
    # vector AP  
    AE = [None, None]  
    AE[0] = E[0] - A[0]  
    AE[1] = E[1] - A[1]   
  
    # Variables to store dot product  
  
    # Calculating the dot product  
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]   
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]   
  
    # Minimum distance from  
    # point E to the line segment  
    reqAns = 0   
  
    # Case 1  
    if (AB_BE > 0) : 
  
        # Finding the magnitude  
        y = E[1] - B[1]   
        x = E[0] - B[0]   
        reqAns = sqrt(x * x + y * y)   
  
    # Case 2  
    elif (AB_AE < 0) : 
        y = E[1] - A[1]   
        x = E[0] - A[0]   
        reqAns = sqrt(x * x + y * y)   
  
    # Case 3  
    else: 
  
        # Finding the perpendicular distance  
        x1 = AB[0]   
        y1 = AB[1]   
        x2 = AE[0]   
        y2 = AE[1]   
        mod = sqrt(x1 * x1 + y1 * y1)   
        reqAns = abs(x1 * y2 - y1 * x2) / mod   
      
    return reqAns

def is_left(A, B, C):
    '''
    Check if point C is to the left of AB.
    '''
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    return np.cross(B[0]-A[0], C[1]-A[1]) - np.cross(B[1]-A[1], C[0]-A[0]) > 0





