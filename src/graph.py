import matplotlib.pyplot as plt
from node import Node
from utils import distance, is_intersect
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
        edges = set()
        for node in self.mapping:
            for neighbor in self.mapping[node]:
                if (node, neighbor) not in edges:
                    edges.add((node, neighbor))
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
        queue.append(start)

        while queue:
            v = queue.popleft()
            if v == end:
                return True

            if length == 0:
                return False

            for neighbor in self.mapping[v]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
            length -= 1

        return False
        

    def draw(self):
        for node in self.mapping:
            plt.scatter(node.x, node.y, c='r')

        for node in self.mapping:
            for target in self.mapping[node]:
                plt.annotate("", xy=(target.x, target.y), xycoords='data', xytext=(node.x, node.y),
                             textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    def delete_node(self, node):
        self.mapping.pop(node, None)
        for other in self.mapping:
            if node in self.mapping[other]:
                self.mapping[other].remove(node)

    def bridge(self):
        for node in self.mapping:
            if len(self.mapping[node]) == 0:
                for target in self.mapping:
                    if target != node and distance(node.x, node.y, target.x, target.y) < 6 and abs(node.heading - target.heading) < 0.4:
                        print("Bridging"+str(node.x)+str(node.y))
                        self.mapping[node].append(target)

    def merge(self):
        to_delete = []
        for node in self.mapping:
            for target in self.mapping:
                if target != node and distance(node.x, node.y, target.x, target.y) < 1.4:
                    print("Merging" + str(node.x) + str(node.y))
                    to_delete.append(target)
                    self.mapping[node] += self.mapping[target]
        for node in to_delete:
            self.delete_node(target)

    def merge_edges(self):
        for edge1 in self.edges():
            for edge2 in self.edges():
                A = [edge1[0].x, edge1[0].y]
                B = [edge1[1].x, edge1[1].y]
                C = [edge2[0].x, edge2[0].y]
                D = [edge2[1].x, edge2[1].y]
                if edge1 != edge2 and is_intersect(A, B, C, D):
                    l1 = LineString([tuple(A), tuple(B)])
                    l2 = LineString([tuple(C), tuple(D)])
                    print(l1.intersection(l2))

        
        
    
    

    