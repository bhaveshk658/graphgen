import os
import random

import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from tqdm import tqdm

import lanelet2 as llt
from lanelet2.projection import UtmProjector

from graphgen.data import coordinate_to_id, gravity, compute_headings
from graphgen.graph import Graph, Node
from graphgen.generate import convert_to_graph

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)



map_path = os.path.abspath('../data/InteractionDR1.0/maps/DR_USA_Roundabout_EP.osm')
print("Path: " + map_path)

projector = UtmProjector(llt.io.Origin(0, 0))
mapLoad, errors = llt.io.loadRobust(map_path, projector)


ids = []
for trace in traces:
    trace_id = []
    for point in trace:
        res = coordinate_to_id(mapLoad, point[0], point[1])
        if len(res) == 0:
            res = float('inf')
        else:
            res.sort()
            res = res[0]
        if len(trace_id) == 0 or res != trace_id[-1]:
            trace_id.append(res)
    ids.append(trace_id)

groups = []
group_ids = []
for i in range(len(traces)):
    trace_id = ids[i]
    trace = traces[i]
    if trace_id in group_ids:
        index = group_ids.index(trace_id)
        groups[index].append(i)
    else:
        group_ids.append(trace_id)
        groups.append([i])

"""
- Compute headings of each point
- Create new list of nodes using point + heading
- Construct graph
"""

traces = compute_headings(traces)
nodes = []
for trace in traces:
    node_trace = [Node(point[0], point[1], point[2]) for point in trace]
    nodes.append(node_trace)

final_nodes = []
for group in groups:
    target_nodes = [nodes[i] for i in group]
    G = convert_to_graph(target_nodes, dist_limit=3)
    final_nodes.append(G.get_lane_nodes())

G = convert_to_graph(final_nodes, dist_limit=2, heading_limit=0.72)
G.draw()






#print(groups)
"""
rand_i = random.randint(0, len(groups)-1)
rand_j = random.randint(0, len(groups)-1)
print(group_ids[rand_i])
print(group_ids[rand_j])
i = groups[rand_i][0]
j = groups[rand_j][0]
print(i, j)
for point in traces[i]:
    plt.scatter(point[0], point[1], c='b', alpha=0.4)
for point in traces[j]:
    plt.scatter(point[0], point[1], c='r', alpha=0.4)
"""

"""
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
color_index = 0
for group in groups:
    if len(group) > 1:
        c = colors[color_index]
        for i in group:
            trace = traces[i]
            for point in trace:
                plt.scatter(point[0], point[1], c=c, alpha=0.2)
        color_index += 1
        color_index = color_index % len(colors)
"""
plt.savefig("graph.png")
