from networkx import convert
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from graphgen.graph import Node, Graph
from graphgen.generate import convert_to_graph

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)
"""
for trace in tqdm(traces):
    for point in trace:
        plt.scatter(point[0], point[1], c='b', alpha=0.1)
"""
traces = [[Node(p[0], p[1], p[2]) for p in trace] for trace in traces]

G = convert_to_graph(traces, dist_limit=2, heading_limit=0.72)
G.draw()

plt.show()

