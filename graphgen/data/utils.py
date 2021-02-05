from math import sqrt, pi, exp

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

def t1_force(t):
    M = 10
    s1 = 5
    s2 = 5
    ss = s1**2 + s2**2
    return (M*t/(ss*sqrt(2*pi*ss))) * exp(-(t**2)/(2*ss))

def t2_force(p, orig):
    return 0.5*dist(p, orig)