import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from graphgen.data import get_training_data, clean, gravity, compute_headings

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_SR"

# Get data within a certain box
print("Fetching data...")
traces = get_training_data([1, 6], path, exclude=True, max_file_num=9)

print("Saving...")
np.save(file="/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_SR/raw_traces.npy", arr=traces)

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
