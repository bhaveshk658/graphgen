import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from graphgen.data import compute_headings
from graphgen.generate import convert_to_graph_nx

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)

for trace in tqdm(traces):
    for point in trace:
        plt.scatter(point[0], point[1], c='b', alpha=0.1)

plt.show()


