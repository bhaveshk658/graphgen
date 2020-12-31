import os
from pandas import read_csv
import numpy as np

from graphgen.data.utils import dist

def get_training_data(n, location):
	'''
	Get training data from n files from string location.
	E.g. get_training_data(4, "path/to/data")
	Returns a list of all traces
	'''
	traces = []

	for i in range(n):
		path = os.path.join(location, "vehicle_tracks_00"+str(i)+".csv")
		data = read_csv(path)

		# Define rectangle of area to take points from.
		box = [[960, 1015], [980, 1040]]
		data = data.loc[(data['x'] > box[0][0]) & (data['x'] < box[0][1]) & (data['y'] > box[1][0]) & (data['y'] < box[1][1])]

		# Add the trace for each car j to the list of traces.
		# Contains x, y, x-velocity, y-velocity.
		for j in range(len(data.index)):
			temp = data.loc[(data['track_id'] == j)]
			temp = temp.to_numpy()
			if len(temp != 0):
				temp = np.vstack((temp[:, 4], temp[:, 5], temp[:, 6], temp[:, 7])).T
				traces.append(temp)

		# Get headings using velocity vector at each point.
		for i in range(len(traces)):
			for j in range(len(traces[i])):
				velocity = traces[i][j][2:]
				length = np.linalg.norm(velocity)
				if (length == 0):
					heading = 0
				else:
					x = velocity/length
					heading = np.arccos(np.dot(x, [1, 0]))
				traces[i][j][3] = heading
		
		# Keep traces as x, y, heading.
		for i in range(len(traces)):
			traces[i] = np.delete(traces[i], 2, axis=1)

	return traces


def clean(traces, length_threshold, dist_threshold):
	"""
	Clean a list of traces, Eliminate traces with length
	less than length_threshold and eliminate points that are
	within dist_threshold of each other.
	"""
	trips = []
	for c in traces:
		# If there are less than length_threshold points, skip this trace.
		if (len(c) < length_threshold):
			continue
		trip = []
		point = c[0]
		for i in range(1, len(c)):
			# If the point is less than dist_threshold unit away, skip it.
			if dist(point, c[i]) < dist_threshold:
				continue
			trip.append([c[i][0], c[i][1], c[i][2]])
			point = c[i]
		trips.append(trip)

	return trips