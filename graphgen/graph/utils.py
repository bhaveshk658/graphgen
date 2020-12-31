
def distance(x1, y1, x2, y2):
    '''
    Distance between two points (x1, y1), (x2, y2).
    '''
    return pow((pow(x1 - x2, 2) + pow(y1 - y2, 2)), 0.5)

def node_dist(node1, node2):
    '''
    Distance between two nodes.
    '''
    return distance(node1.x, node1.y, node2.x, node2.y)