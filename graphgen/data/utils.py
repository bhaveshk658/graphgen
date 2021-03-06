from math import sqrt, pi, exp
import numpy as np

def distance(x1, y1, x2, y2):
    """
    Distance between two points (x1, y1), (x2, y2).
    """
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def array_dist(p1, p2):
    """
    Distance between two points represented by arrays.
    """
    return distance(p1[0], p1[1], p2[0], p2[1])

def direction(p1, p2):
    """
    Direction between two points p1 and p2.
    """
    return (p2-p1)/distance(p1[0], p1[1], p2[0], p2[1])

def theta(v1, v2):
    """
    Computes angle between two vectors
    """
    val = v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    if val > 1:
        print(val, v1, v2)
        return np.arccos(1)
    elif val < -1:
        print(val, v1, v2)
        return np.arccos(-1)
    return np.arccos(val)

def proj(v1, v2):
    """
    Computes projection of v1 onto v2
    """
    return np.dot(v1, v2)/np.linalg.norm(v2)

def t1_force(t):
    M = 0.01
    N = 15
    s1 = 2
    s2 = 0.3
    sig_square = s1**2 + s2**2

    s = s1**2 + s2**2
    A = -(M * N) / sqrt(2*pi*s)
    return A * exp(-(t**2) / (2*s)) * (-t/s)
    #return (M*N*t/(ss*sqrt(2*pi*ss))) * exp(-(t**2)/(2*ss))
    #return ((M*N)/(sqrt(2*pi*sig_square))) * ((2*t*exp((-t**2)/(2*sig_square)))/(2*sig_square))

def t2_force(p, orig):
    return 0.005*array_dist(p, orig)

def ccw(A, B, C):
    """
    Check if a point C is counter-clockwise to AB.
    """
    return (C[1] - A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def is_intersect(A, B, C, D):
    """
    Check if two line segments AB and CD intersect.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def line_line_segment_intersect(p, d, p1, p2):
    """
    Check if a line defined by point p and direction d intersects
    a segment p1p2.
    """

    extend1 = [p[0] + 500000*d[0], p[1] + 500000*d[1]]
    extend2 = [p[0] - 500000*d[0], p[1] - 500000*d[1]]

    return is_intersect(p, extend1, p1, p2) or is_intersect(p, extend2, p1, p2)

def minDistance(A, B, E) :
    """
    Minimum distance from a point E to line segment AB. Adapted from GeeksforGeeks.
    """

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
