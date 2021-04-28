"""
import lanelet2 as llt
from lanelet2.projection import UtmProjector
import matplotlib.pyplot as plt
import os
import numpy as np
from graphgen.data import coordinate_to_id, draw_map, gravity, compute_headings

map_path = os.path.abspath('../data/InteractionDR1.0/maps/DR_USA_Roundabout_EP.osm')
traces = np.load(file="traces.npy", allow_pickle=True)

draw_map(map_path)
for trace in traces:
    for point in trace:
        plt.scatter(point[0], point[1], c='b', alpha=0.1)
"""
from graphgen.data import get_training_data, clean
import matplotlib.pyplot as plt

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

print("Fetching...")
traces = get_training_data(2, path)
traces = clean(traces, 50, 2)
print("Plotting...")
for trace in traces:
    for point in trace:
        plt.scatter(point[0], point[1], c='b', alpha=0.1)
print("Showing...")
plt.show()