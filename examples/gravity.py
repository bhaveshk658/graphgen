import graphgen
from graphgen.data import get_training_data, clean, gravity
from graphgen.graph import Graph, Node
from graphgen.generate import convert_to_graph

import matplotlib.pyplot as plt

# Get data
path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

# Get data within a certain box
print("Fetching data...")
traces = get_training_data(1, path, xmin=960, xmax=1015, ymin=980, ymax=1040)

print("Cleaning data...")
traces = clean(traces, length_threshold=50, dist_threshold=1)

# All traces going from right to bottom
rb = [0, 1, 8, 11, 14, 15, 18, 19, 21, 24]
rb_traces = [traces[i] for i in rb]


print("Plotting initial points...")
plt.figure(1)
plt.xlim(960, 1015)
plt.ylim(980, 1040)
for trace in rb_traces:
    for point in trace:
        plt.scatter(point[0], point[1], c='r', alpha=0.2)


# Perform preprocessing and plot
print("Preprocessing...")
processed_rb_traces = gravity(rb_traces)


print("Plotting preprocessed points...")
plt.figure(3)
plt.xlim(960, 1015)
plt.ylim(980, 1040)
for trace in processed_rb_traces:
    for point in trace:
        plt.scatter(point[0], point[1], c='r', alpha=0.2)

plt.show()
