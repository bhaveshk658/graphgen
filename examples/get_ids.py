import os
import random

import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from tqdm import tqdm

import lanelet2 as llt
from lanelet2.projection import UtmProjector

from graphgen.data import coordinate_to_id, draw_map, gravity, compute_headings
from graphgen.graph import Graph, Node
from graphgen.generate import convert_to_graph

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)



map_path = os.path.abspath('../data/InteractionDR1.0/maps/DR_USA_Roundabout_EP.osm')
draw_map(map_path)

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


plt.savefig("graph.png")
