import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from math import hypot

import os
import time

from interact_toolset import distance

def get_training_data(k, location):
	'''
	Get training data from k files from string location.
	E.g. get_training_data(4, "DR_USA_Roundabout_EP")
	Returns a dataframe of k files compiled together.
	'''
	frames = []

	for i in range(k):
		data = pd.read_csv(os.path.join("interaction-dataset-copy/recorded_trackfiles/"
			+location, "vehicle_tracks_00"+str(k)+".csv"))
		frames.append(data)

	return pd.concat(frames)

def 



