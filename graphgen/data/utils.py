
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