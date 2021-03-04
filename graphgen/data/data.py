import os
import time
import random

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
			cleaned_trace.append([trace[i][0], trace[i][1], trace[i][2]])
			point = trace[i]
		cleaned_traces.append(cleaned_trace)

	return cleaned_traces

def gravity(traces):
	"""
	Given a list of traces, perform Cao & Krumm preprocessing.
	"""
	lengths = [len(trace) for trace in traces]
	# Flatten to list of points, keep copy of original positions
	points = np.array([item for sublist in traces for item in sublist])
	headings = points[:, 2]
	points = points[:, :2]
	original = np.copy(points)


	tree = KDTree(points, leaf_size=2)

	rand_index = random.randrange(1, len(points)-1, 1)
	# Number of iterations
	for k in range(5):

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
			if (i == rand_index):
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
				if i == rand_index:
					plt.scatter(b[0], b[1], c='g', alpha=0.4)
					plt.scatter(c[0], c[1], c='g', alpha=0.4)
				"""

				# Compute type 1 force (gravitational)
				midpoint = (c+b)/2
				t1_distance = array_dist(a, midpoint)
				t1_direction = direction(a, midpoint)
				t1 = np.array(t1_force(t1_distance) * t1_direction)


				# Compute type 2 force (spring)
				if all(a == orig):
					t2 = 0
				else:
					t2_direction = direction(a, orig)
					t2 = t2_force(a, orig) * t2_direction

				resultant = t1 + t2

				#if i == rand_index:
				a += (4*resultant)#3.7 is the number

				res_mag = pow(resultant[0]**2 + resultant[1]**2, 0.5)
				unit_rest= resultant / res_mag
				resultants[i] += resultant

			"""
			if i == rand_index:
				plt.scatter(a[0], a[1], c='y')
			"""

			points[i] = a
			# Update kd tree
			tree = KDTree(points, leaf_size=2)

		"""
		# Get max mag of resultants
		max_res = max(resultants, key=lambda v: pow(v[0]**2+ v[1]**2, 0.5))
		print("Max resultant: " + str(pow(max_res[0]**2 + max_res[1]**2, 0.5)))

		# Shift all points by resultants
		for i in range(len(points)):
			points[i] += resultants[i]

		# Plot points
		plt.figure(k+2)
		plt.xlim(960, 1015)
		plt.ylim(980, 1040)
		for point in points:
			plt.scatter(point[0], point[1], c='r', alpha=0.2)

		# Update KD tree
		tree= KDTree(points, leaf_size=2)
		"""

	new_traces = []
	for l in lengths:
		trace = []
		temp = points[:l]
		temp_headings = headings[:l]
		for i in range(len(temp)):
			h = temp_headings[i]
			trace.append([temp[i][0], temp[i][1], h])
		new_traces.append(trace)
		points = points[l:]
		headings = headings[l:]
	return new_traces
