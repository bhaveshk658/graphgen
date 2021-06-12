import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graphgen.generate import convert_to_graph_nx, get_lane_points, convert_lanes_to_graph, clean

map_dir = "/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_EP/"

with open(map_dir + "groups.x", 'rb') as f:
    groups = pickle.load(f)

print("Loading...")
traces = np.load(file=map_dir + "traces.npy", allow_pickle=True)

print("Converting lanes...")
lanes = []
for group in groups.values():
    target_nodes = [traces[i] for i in group]
    G = convert_to_graph_nx(target_nodes)
    lanes.append(get_lane_points(G))

# print("Plotting without cleaning...")
# plt.figure(1)
# G = convert_lanes_to_graph(lanes, dist_limit=2, heading_limit=0.785398)
# nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10)

# print("Plotting with cleaning...")
# plt.figure(2)
G = convert_lanes_to_graph(lanes, dist_limit=2, heading_limit=0.53)
G = clean(G, 7)
# nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10)
# plt.show()

with open(map_dir + "graph.x", "wb") as f:
    pickle.dump(G, f)
