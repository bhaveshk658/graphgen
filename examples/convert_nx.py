import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphgen.generate import convert_to_graph_nx, get_lane_points, convert_lanes_to_graph, clean

with open('groups.x', 'rb') as f:
    groups = pickle.load(f)

print("Loading...")
traces = np.load(file="traces3.npy", allow_pickle=True)

lanes = []
for group in groups.values():
    target_nodes = [traces[i] for i in group]
    G = convert_to_graph_nx(target_nodes)
    lanes.append(get_lane_points(G))

plt.figure(1)
G = convert_lanes_to_graph(lanes, dist_limit=2, heading_limit=0.785398)
# clean(G, 3)
nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10)

plt.figure(2)
G = convert_lanes_to_graph(lanes, dist_limit=2, heading_limit=0.785398)
G = clean(G, 2)
nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10)
plt.show()
