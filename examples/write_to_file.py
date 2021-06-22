import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from graphgen.data import get_training_data, clean, gravity, compute_headings

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_DEU_Merging_MT"

# Get data within a certain box
print("Fetching data...")
traces = get_training_data([0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13], path)

print("Saving...")
np.save(file="/Users/bkalisetti658/desktop/graphgen/data/DR_DEU_Merging_MT/raw_traces.npy", arr=traces)

# print("Cleaning data...")
# traces = clean(traces, length_threshold=50, dist_threshold=2)

# print("Saving...")
# np.save(file="/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_EP/traces2.npy", arr=traces)

# traces = gravity(traces, resultant_threshold=0.3)

# print("Saving...")
# np.save(file="/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_EP/traces3.npy", arr=traces)

# print("Computing headings")
# traces = compute_headings(traces)

# print("Saving...")
# np.save(file="/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_EP/traces.npy", arr=traces)
