import numpy as np
import matplotlib.pyplot as plt

from graphgen.data import compute_headings
from graphgen.graph import Graph, Node
from graphgen.generate import convert_to_graph

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)

traces = compute_headings(traces)

nodes = []
for trace in traces:
    node_trace = [Node(point[0], point[1], point[2]) for point in trace]
    nodes.append(node_trace)

G = convert_to_graph(nodes)
G.draw()
plt.savefig("no_lanelet_graph.png")
