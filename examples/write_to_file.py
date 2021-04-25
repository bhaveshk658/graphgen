import numpy as np
import matplotlib.pyplot as plt

from graphgen.data import get_training_data, clean, gravity#, compute_headings

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

# Get data within a certain box
print("Fetching data...")
traces = get_training_data(2, path)

print("Cleaning data...")
traces = clean(traces, length_threshold=50, dist_threshold=2)

traces = gravity(traces)

print("Saving...")
np.save(file="traces", arr=traces)
