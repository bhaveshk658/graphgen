import os
import time
import random
from math import cos

from pandas import read_csv
import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from graphgen.data.utils import *
from graphgen.graph import Node

MAX_NUM = float('inf')
MIN_NUM = -float('inf')

def get_training_data(n, location, box=None, xmin=MIN_NUM, xmax=MAX_NUM, ymin=MIN_NUM, ymax=MAX_NUM):
	"""
	Get training data from n files from string location.
	E.g. get_training_data(4, "path/to/data", xmin, xmax, ymin, ymax)
	Returns a list of all traces
	"""
	traces = []

	for file_num in range(n):
		path = os.path.join(location, "vehicle_tracks_00"+str(file_num)+".csv")
		data = read_csv(path)

		# Define rectangle of area to take points from.
		data = data.loc[(data['x'] > xmin) & (data['x'] < xmax) & (data['y'] > ymin) & (data['y'] < ymax)]

		# Add the trace for each car j to the list of traces.
		# Contains x, y, x-velocity, y-velocity.
		for i in range(len(data.index)):
			temp = data.loc[(data['track_id'] == i)]
			temp = temp.to_numpy()
			if len(temp != 0):
				temp = np.vstack((temp[:, 4], temp[:, 5])).T
				traces.append(temp)

	return np.array(traces, dtype="object")


def clean(traces, length_threshold, dist_threshold):
	"""
	Clean a list of traces, Eliminate traces with length
	less than length_threshold and eliminate points that are
	within dist_threshold of each other.
	"""
	cleaned_traces = []
	for trace in traces:
		# If there are less than length_threshold points, skip this trace.
		if (len(trace) < length_threshold):
			continue
		cleaned_trace = []
		point = trace[0]
		for i in range(1, len(trace)):
			# If the point is less than dist_threshold unit away, skip it.
			if array_dist(point, trace[i]) < dist_threshold:
				continue
			cleaned_trace.append([trace[i][0], trace[i][1]])
			point = trace[i]
		cleaned_traces.append(np.array(cleaned_trace))

	return np.array(cleaned_traces, dtype="object")

def gravity(traces):
	"""
	Given a list of traces, perform Cao & Krumm preprocessing.
	"""
	lengths = [len(trace) for trace in traces]
	# Flatten to list of points, keep copy of original positions
	points = np.array([item for sublist in traces for item in sublist])
	original = np.copy(points)

	tree = KDTree(points, leaf_size=2)

	rand_index= random.randrange(1, len(points)-1, 1)
	# Number of iterations
	k = 0
	resultant_threshold = 0.1
	repeat = True
	print("Iteration will stop when resultant is less than " + str(resultant_threshold))
	print("Processing " + str(len(points)) + " points (" + str(len(traces)) + ") traces")
	while repeat:
		k += 1
		print("Starting iteration " + str(k) + "...")

		# Initialize resultants array
		resultants = [[0, 0]]*len(points)
		# Skip first and last points
		for i in range(1, len(points) - 1):
			a = points[i]
			orig = original[i]

			# Get perpendicular heading
			prev = points[i-1]
			next = points[i+1]
			heading = next - prev
			d = [heading[1], -heading[0]]

			# Query for nearby points
			ind = tree.query_radius(np.array([a]), r=2)
			ind[0].sort()

			# Identify all edges by finding consecutive nearby points
			# Check for incomplete edges, pair not coming from same trace?
			pairs = []
			for j in range(1, len(ind[0])):
				if ind[0][j] == ind[0][j-1] + 1:
					pairs.append((ind[0][j-1], ind[0][j]))

			# Debugging
			"""
			if (i == rand_index and k == num_iter-1):
				plt.figure(3)
				plt.scatter(orig[0], orig[1], c='m') # Original location
				plt.scatter(a[0], a[1], c='b') # Point
				plt.scatter(points[i-1][0], points[i-1][1], c='b') # Point before
				plt.scatter(points[i+1][0], points[i+1][1], c='b') # Point after
				plt.scatter(a[0] + d[0], a[1] + d[1], c='k') # Heading
				plt.scatter(a[0]-d[0], a[1]-d[1], c='k') # Heading
			"""

			# Iterate through all edges, identifying intersecting ones
			intersecting_pairs = []
			for pair in pairs:
				b = points[pair[0]]
				c = points[pair[1]]
				midpoint = (b+c)/2
				# If edge intersects perpendicular line from point, proceed
				if line_line_segment_intersect(a, d, b, c):
					intersecting_pairs.append(pair)

			# Iterate through intersecting edges
			for pair in intersecting_pairs:
				b = points[pair[0]]
				c = points[pair[1]]

				"""
				if i == rand_index and k == num_iter-1:
					plt.figure(3)
					plt.scatter(b[0], b[1], c='g', alpha=0.4)
					plt.scatter(c[0], c[1], c='g', alpha=0.4)
				"""

				# Compute type 1 force (gravitational)
				midpoint = (c+b)/2
				t1_distance = array_dist(a, midpoint)
				t1_direction = direction(a, midpoint)
				t1 = np.array(t1_force(t1_distance) * t1_direction)

				# Multiply t1 force by cosine of angle between edge and heading of point
				angle = theta(np.array(heading), np.array([c[0]-b[0], c[1]-b[1]]))
				t1 *= cos(angle)


				# Compute type 2 force (spring)
				if all(a == orig):
					t2 = 0
				else:
					t2_direction = direction(a, orig)
					t2 = t2_force(a, orig) * t2_direction

				resultant = t1 + t2

				res_mag = pow(resultant[0]**2 + resultant[1]**2, 0.5)
				unit_res = resultant / res_mag
				resultants[i] += resultant

			"""
			if i == rand_index and k == num_iter-1:
				plt.figure(3)
				plt.scatter(a[0] + resultants[i][0], a[1] + resultants[i][1], c='y')
			"""

		# Move all points
		for i in range(len(points)):
			points[i] += resultants[i]

		# Recreate k-d tree with new points
		tree = KDTree(points, leaf_size=2)


		# Get max mag of resultants
		max_res = max(resultants, key=lambda v: pow(v[0]**2+ v[1]**2, 0.5))
		max_res_mag = pow(max_res[0]**2 + max_res[1]**2, 0.5)
		print("Max resultant: " + str(max_res_mag))
		if max_res_mag < 0.1:
			repeat = False


	# Recreate traces using stored lengths
	new_traces = []
	for l in lengths:
		trace = []
		temp = points[:l]
		for i in range(len(temp)):
			trace.append([temp[i][0], temp[i][1]])
		new_traces.append(trace)
		points = points[l:]
	return new_traces
