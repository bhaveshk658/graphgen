import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, atan2, pi
from dipy.segment.metric import Metric, ResampleFeature
from dipy.segment.clustering import QuickBundles

from graphgen.graph import Graph, Node
from graphgen.generate import convert_to_graph
from graphgen.data.utils import direction, theta

traces = np.load("trace.npy", allow_pickle=True)

nb_points = len(max(traces, key=lambda x: len(x)))



class TestDistance(Metric):
    def __init__(self):
        super(TestDistance, self).__init__(feature=ResampleFeature(nb_points=nb_points))

    def are_compatible(self, shape1, shape2):
        return len(shape1) == len(shape2)

    def dist(self, v1, v2):
        x = [sqrt((p[0][0]-p[1][0])**2 + (p[0][1]-p[1][1])**2) for p in list(zip(v1, v2))]
        d = np.mean(x)
        return d

print("Clustering...")
metric = TestDistance()
qb = QuickBundles(threshold=3, metric=metric)
clusters = qb.cluster(traces)


print("Computing headings...")
headings = []
for trace in traces:
    trace_headings = []

    dir_vector = direction(trace[0], trace[1])
    angle = atan2(dir_vector[1], dir_vector[0])
    if angle < 0:
        angle += 2*pi
    trace_headings.append(angle)

    for i in range(1, len(trace) - 1):
        prev = trace[i-1]
        next = trace[i+1]
        dir_vector = direction(np.array([prev[0], prev[1]]), np.array([next[0], next[1]]))
        angle = atan2(dir_vector[1], dir_vector[0])
        if angle < 0:
            angle += 2*pi
        trace_headings.append(angle)

    dir_vector = direction(trace[-2], trace[-1])
    angle = atan2(dir_vector[1], dir_vector[0])
    if angle < 0:
        angle += 2*pi
    trace_headings.append(angle)


    headings.append(np.array([trace_headings]))

for i in range(len(traces)):
    traces[i] = np.concatenate((traces[i], headings[i].T), axis=1)

node_traces = []
for trace in traces:
    temp = []
    for point in trace:
        node = Node(point[0], point[1], point[2])
        temp.append(node)
    node_traces.append(temp)

lanes = []
for cluster in 

"""
G = convert_to_graph(traces_with_headings, dist_limit=2)
G.draw()
plt.show()
"""
print(clusters[0][0][0])
print(traces[0][0])
