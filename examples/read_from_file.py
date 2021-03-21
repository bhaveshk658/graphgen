import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)

print("Plotting...")
for i in tqdm(range(len(traces))):
    for point in traces[i]:
        plt.scatter(point[0], point[1], c='b', alpha=0.3)

print("Drawing...")
plt.show()
