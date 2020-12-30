import matplotlib.pyplot as plt
from node import Node
from utils import distance, is_intersect, edge_heading
from shapely.geometry import LineString
from collections import deque
from sklearn.neighbors import KDTree
import numpy as np

class Graph:

    def __init__(self, mapping=None):
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = dict()
        self.nodes = []
        self.points = []
        self.kd = None

    def get_nodes(self):
        return list(self.mapping.keys())

    def get_points(self):
        nodes = self.get_nodes()
        for i in range(len(nodes)):
            node = nodes[i]
            nodes[i] = [node.x, node.y, node.heading]
        return nodes

    def get_points_and_nodes(self):
        nodes = self.get_nodes()
        points = [None]*len(nodes)
        for i in range(len(nodes)):
            node = nodes[i]
            points[i] = [node.x, node.y, node.heading]
        return nodes, points

    def edges(self):
        edges = []
        for node in self.mapping:
            for neighbor in self.mapping[node]:
                if (node, neighbor) not in edges:
                    edges.append((node, neighbor))
        return edges
    
    def add_node(self, node):
        if node not in self.mapping:
            self.mapping[node] = []
            self.nodes.append(node)
            self.points.append([node.x, node.y, node.heading])
    
    def add_edge(self, start, end):
        if start in self.mapping:
            self.mapping[start].append(end)
        else:
            self.mapping[start] = [end]
        if end not in self.mapping:
            self.mapping[end] = []
            self.nodes.append(end)
            self.points.append([end.x, end.y, end.heading])

    def has_path(self, start, end, length):
        visited = set()
        visited.add(start)

        queue = deque()
        queue.append((start, 0))

        while queue:
            v, distance = queue.popleft()
            if v == end:
                return True

            if distance > length:
                return False

            for neighbor in self.mapping[v]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
                    visited.add(neighbor)

        return False
        

    def draw(self):
        for node in self.mapping:
            plt.scatter(node.x, node.y, c='b')

        for node in self.mapping:
            for target in self.mapping[node]:
                plt.annotate("", xy=(target.x, target.y), xycoords='data', xytext=(node.x, node.y),
                             textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    def delete_node(self, node):
        self.mapping.pop(node, None)
        self.nodes.remove(node)
        self.points.remove([node.x, node.y, node.heading])
        for other in self.mapping:
            if node in self.mapping[other]:
                self.mapping[other].remove(node)

    def delete_edge(self, start, end):
        self.mapping[start].remove(end)

    def update(self, G):
        self.mapping.update(G.mapping)
        self.points += G.points
        self.nodes += G.nodes

        
    def cleanup(self):
        for edge in self.edges():
            start = edge[0]
            end = edge[1]
            if abs(edge_heading(start, end) - start.heading) > 0.4:
                if len(self.mapping[start]) > 1:
                    self.mapping[start].remove(end)
            #Compare heading between nodes to heading of start node. If too far off, delete edge.


    def second_order_cleanup(self):
        for edge in self.edges():
            first = edge[0]
            second = edge[1]
            if len(self.mapping[second]) > 1:
                for third in self.mapping[second]:
                    if len(self.mapping[third]) == 1:
                        fourth = self.mapping[third][0]
                        if abs(edge_heading(second, third) - edge_heading(third, fourth)) > 0.5:
                            self.mapping[second].remove(third)


    def get_kd(self):
        if self.kd:
            return self.kd
        points = self.get_points()
        tree = KDTree(np.array(self.points))
        self.kd = tree
        return tree


    def match_point(self, point):
        print("Starting matching")
        tree = self.get_kd()
        print("Got tree")
        dist, ind = tree.query(point)
        print("Query done")
        print(ind)
        return self.nodes[ind[0][0]], self.points[ind[0][0]]

    def match_trace(self, trace):
        tree = self.get_kd()
        node_path = []
        point_path = []
        for point in trace:
            dist, ind = tree.query([point])
            node_path.append(self.nodes[ind[0][0]])
            point_path.append(self.points[ind[0][0]])
        return node_path, point_path


    