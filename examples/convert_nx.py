import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphgen.generate import convert_to_graph_nx, get_lane_points

with open('groups.x', 'rb') as f:
    groups = pickle.load(f)

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)

lanes = []
for group in groups:
    target_nodes = [traces[i] for i in group]
    G = convert_to_graph_nx(target_nodes)
    lanes.append(get_lane_points(G))


G = convert_to_graph_nx(lanes, dist_limit=2, heading_limit=0.785398)
nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10)
plt.show()
