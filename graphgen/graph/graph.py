import matplotlib.pyplot as plt
from shapely.geometry import LineString
from collections import deque
from sklearn.neighbors import KDTree
import numpy as np

from graphgen.graph.node import Node

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
        """
        Return a list of all nodes in the graph.
        """
        return list(self.mapping.keys())

    def get_points(self):
        """
        Return a 2D array of all points (including headings) in the graph.
        """
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
        """
        Return a list of tuples representing all edges in the graph.
        """
        edges = []
        for node in self.mapping:
            for neighbor in self.mapping[node]:
                if (node, neighbor) not in edges:
                    edges.append((node, neighbor))
        return edges
    
    def add_node(self, node):
        """
        Add a node to the graph
        """
        if node not in self.mapping:
            self.mapping[node] = []
            self.nodes.append(node)
            self.points.append([node.x, node.y, node.heading])
    
    def add_edge(self, start, end):
        """
        Add an edge to the graph.
        """
        if start in self.mapping:
            self.mapping[start].append(end)
        else:
            self.mapping[start] = [end]
        if end not in self.mapping:
            self.mapping[end] = []
            self.nodes.append(end)
            self.points.append([end.x, end.y, end.heading])

    def has_path(self, start, end, length):
        """
        Check if a path shorter than 'length' exists from start to end.
        """
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
        """
        Scatter plot the graph.
        """
        for node in self.mapping:
            plt.scatter(node.x, node.y, c='b')

        for node in self.mapping:
            for target in self.mapping[node]:
                plt.annotate("", xy=(target.x, target.y), xycoords='data', xytext=(node.x, node.y),
                             textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    def delete_node(self, node):
        """
        Delete a node from the graph.
        """
        self.mapping.pop(node, None)
        self.nodes.remove(node)
        self.points.remove([node.x, node.y, node.heading])
        for other in self.mapping:
            if node in self.mapping[other]:
                self.mapping[other].remove(node)

    def delete_edge(self, start, end):
        """
        Delete an edge from the graph.
        """
        self.mapping[start].remove(end)

    def update(self, G):
        """
        Merge graph G into the graph.
        """
        self.mapping.update(G.mapping)
        self.points += G.points
        self.nodes += G.nodes

    def get_kd(self):
        """
        Get the graph's k-d tree, or make a new one.
        """
        if self.kd:
            return self.kd
        tree = KDTree(np.array(self.points))
        self.kd = tree
        return tree


    def match_point(self, point):
        """
        Match a point to a node on the graph.
        """
        print("Starting matching")
        tree = self.get_kd()
        print("Got tree")
        _, ind = tree.query(point)
        print("Query done")
        print(ind)
        return self.nodes[ind[0][0]], self.points[ind[0][0]]

    def match_trace(self, trace):
        """
        Match a trace to a path of nodes on the graph.
        """
        tree = self.get_kd()
        node_path = []
        point_path = []
        for point in trace:
            dist, ind = tree.query([point])
            node_path.append(self.nodes[ind[0][0]])
            point_path.append(self.points[ind[0][0]])
        return node_path, point_path


    