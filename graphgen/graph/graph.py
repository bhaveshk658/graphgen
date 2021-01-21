import matplotlib.pyplot as plt
from shapely.geometry import LineString
from collections import deque
from sklearn.neighbors import KDTree
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from graphgen.graph.node import Node
from graphgen.graph.utils import *

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
        return self.nodes

    def get_points(self):
        """
        Return a 2D array of all points (including headings) in the graph.
        """
        nodes = self.get_nodes()
        for i in range(len(nodes)):
            node = nodes[i]
            nodes[i] = [node.x, node.y, node.heading]
        return nodes

    def get_lane_nodes(self):
        """
        Get nodes of a lane in order.
        """
        start = None
        end_nodes = np.array(self.edges())[:, 1]
        for node in self.nodes:
            if node not in end_nodes:
                start = node
                break
        nodes = []
        while start:
            nodes.append(start)
            if len(self.mapping[start]) != 0:
                start = self.mapping[start][0]
            else:
                start = None
        return nodes
    def get_lane_points(self):
        """
        Get points of a lane in order.
        """
        start = None
        end_nodes = np.array(self.edges())[:, 1]
        for node in self.nodes:
            if node not in end_nodes:
                start = node
                break
        points = []
        while start:
            points.append([start.x, start.y])
            if len(self.mapping[start]) != 0:
                start = self.mapping[start][0]
            else:
                start = None
        return points


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
        """
        Find start node of G (G should be a specific lane/path)
        For node starting at start:
            Query nearest edge in self
            d = distance from edge to n
            If d < threshold:
                merge node into edge
            else:
                add node to graph

        """
        # empty graph check
        if not self.mapping:
            self.mapping.update(G.mapping)
            self.nodes += G.nodes
            self.points += G.points
            return
        """
        start = None
        end_nodes = np.array(G.edges())[:, 1]
        for node in G.nodes:
            if node not in end_nodes:
                start = node
                break

        node_edge_map = dict()
        point = start
        i = 0
        while point:
            if len(G.mapping[point]) != 0:
                next = G.mapping[point][0]
            else:
                next = None
            d, edge = self.get_nearest_edge(point)
            heading_diff = abs(point.heading - edge[0].heading)
            if d < 0 and heading_diff < 0.2:
                
                node_edge_map[point] = edge
            point = next
            i += 1

        i = 0
        while start:
            if len(G.mapping[start]) != 0:
                next = G.mapping[start][0]
            else:
                next = None
            if start in node_edge_map:
                edge = node_edge_map[start]
                print(i, start.x, start.y, edge[0].x, edge[0].y, edge[1].x, edge[1].y)
                #if edge[1] in self.mapping[edge[0]]:
                self.mapping[edge[0]].remove(edge[1])
                self.mapping[edge[0]].append(start)
                self.mapping[start] = [edge[1]]
            else:
                self.mapping[start] = [next] if next else []
            start = next
            i += 1
        """
        self_nodes = self.get_lane_nodes()
        G_nodes = G.get_lane_nodes()
        x_start, x_end, y_start, y_end = self.id_merge_region(G)
        while x_start <= x_end and y_start <= y_end:
            node = self_nodes[x_start]
            plt.scatter(node.x, node.y, c='g')
            x_start += 1

            node = G_nodes[y_start]
            plt.scatter(node.x, node.y, c='g')
            y_start += 1
        
        while x_start <= x_end:
            node = self_nodes[x_start]
            plt.scatter(node.x, node.y, c='g')
            x_start += 1
        
        while y_start <= y_end:
            node = G_nodes[y_start]
            plt.scatter(node.x, node.y, c='g')
            y_start += 1

        

    def get_kd(self):
        """
        Get the graph's k-d tree, or make a new one.
        """
        """
        if self.kd:
            return self.kd
        """
        tree = KDTree(np.array(self.points))
        self.kd = tree
        return tree


    def match_point(self, point):
        """
        Match a point to a node on the graph.
        """
        tree = self.get_kd()
        _, ind = tree.query([point])
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

    def merge_nodes(self, node1, node2):
        """
        Merge two nodes together.
        """
        new_x = (node1.x + node2.x) / 2
        new_y = (node1.y + node2.y) / 2
        new_heading = -1

        node1.x = new_x
        node1.y = new_y
        node1.heading = new_heading

        if node1 in self.mapping and node2 in self.mapping:
            for node in self.mapping[node2]:
                if node not in self.mapping[node1]:
                    self.mapping[node1].append(node)

            del self.mapping[node2]

            self.nodes = self.mapping.keys()
            self.points = [[node.x, node.y, node.heading] for node in self.nodes]

            nodes = self.get_nodes()
            for prev in nodes:
                if node2 in self.mapping[prev]:
                    self.mapping[prev].remove(node2)
                    self.mapping[prev].append(node1)
                    break
        

    def cleanup(self):
        """
        Cleanup a graph by merging similar nodes together
        """
        nodes = self.get_nodes()
        for node1 in nodes:
            for node2 in nodes:
                if node1 != node2 and node_dist(node1, node2) < 1:
                    self.merge_nodes(node1, node2)

    def get_nearest_edge(self, node):
        """
        Finds edge closest to node.
        """
        edges = self.edges()
        min_dist = float('inf')
        closest_edge = None
        for edge in edges:
            d = edge_dist(edge[0], edge[1], node)
            if d < min_dist:
                min_dist = d
                closest_edge = edge
        return min_dist, closest_edge

    def id_merge_region(self, G):
        """
        Finds portion of both graphs to merge together.
        """
        self_points = np.array(self.get_lane_points())
        G_points = np.array(G.get_lane_points())
        
        _, path = fastdtw(self_points, G_points)

        x_ind = []
        y_ind = []
        for pair in path:
            x = self_points[pair[0]]
            y = G_points[pair[1]]
            if distance(x[0], x[1], y[0], y[1]) < 1.75:
                x_ind.append(pair[0])
                y_ind.append(pair[1])
        
        x_start = min(x_ind)
        x_end = max(x_ind)

        y_start = min(y_ind)
        y_end = max(y_ind)

        """
        for i in range(x_start, x_end + 1):
            plt.scatter(self_points[i][0], self_points[i][1], c='g')
        for i in range(y_start, y_end + 1):
            plt.scatter(G_points[i][0], G_points[i][1], c='g')

        """

        return x_start, x_end, y_start, y_end
        



    