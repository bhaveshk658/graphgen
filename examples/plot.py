from networkx import convert
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from graphgen.graph import Node, Graph
from graphgen.generate import convert_to_graph_nx, clean

import networkx as nx

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)
print(len(traces), sum([len(trace) for trace in traces]))
# """
# for trace in tqdm(traces):
#     for point in trace:
#         plt.scatter(point[0], point[1], c='b', alpha=0.1)
# """
# #traces = [[Node(p[0], p[1], p[2]) for p in trace] for trace in traces]

# G = convert_to_graph_nx_real(traces, dist_limit=2, heading_limit=0.72)
# G = clean(G, 3)
# nx.draw(G, nx.get_node_attributes(G, 'pos'), node_size=10)

# plt.show()


