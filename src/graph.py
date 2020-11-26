import matplotlib.pyplot as plt
from node import Node
from utils import distance, is_intersect, edge_heading
from shapely.geometry import LineString
from collections import deque

class Graph:

    def __init__(self, mapping=None):
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = dict()

    def get_nodes(self):
        return list(self.mapping.keys())

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
    
    def add_edge(self, start, end):
        if start in self.mapping:
            self.mapping[start].append(end)
        else:
            self.mapping[start] = [end]
        if end not in self.mapping:
            self.mapping[end] = []

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
        for other in self.mapping:
            if node in self.mapping[other]:
                self.mapping[other].remove(node)

    def delete_edge(self, start, end):
        self.mapping[start].remove(end)

        
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


    