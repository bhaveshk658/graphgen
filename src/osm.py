import osmnx
import networkx as nx
import matplotlib.pyplot as plt

from node import Node
from graph import Graph

G = Graph()
box = [[960, 1015], [980, 1040]]


path = "interaction-dataset-copy/maps/DR_USA_Roundabout_EP.osm"
'''
f = open(path, 'r')

i = 0
for line in f:
    if i == 619:
        break
    if i != 0:
        line = line.split()
        x = float(line[2][3:-1])
        y = float(line[3][3:-1])
        if x > box[0][0] and x < box[0][1] and y > box[1][0] and y < box[1][1]:
            node = Node(x, y, 0)
            G.add_node(node)
    i += 1

G.draw()
plt.show()
'''

g = osmnx.graph_from_xml(path)
nx.drawing.draw_networkx(g)
plt.show()


