from networkx import convert
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Loading...")
traces = np.load(file="/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_EP/traces.npy", allow_pickle=True)
# print(len(traces), sum([len(trace) for trace in traces]))

for trace in tqdm(traces):
    for point in trace:
        plt.scatter(point[0], point[1], c='b', alpha=0.1)

plt.show()


