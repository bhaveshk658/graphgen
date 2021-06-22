import os
import time
import random
from math import cos, atan2, sqrt, pi

from pandas import read_csv
import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from graphgen.data.utils import *
from graphgen.graph import Node

MAX_NUM = float('inf')
MIN_NUM = -float('inf')

def get_training_data(file_nums, location, xmin=MIN_NUM, xmax=MAX_NUM, ymin=MIN_NUM, ymax=MAX_NUM):
	"""
	Get training data from n files from string location.
	E.g. get_training_data(4, "path/to/data", xmin, xmax, ymin, ymax)
	Returns a list of all traces
	"""
	traces = []

	if isinstance(file_nums, int):
		file_nums = range(file_nums)


	for file_num in file_nums:
		print("Processing file " + str(file_num))
		if file_num < 10:
			path = os.path.join(location, "vehicle_tracks_00"+str(file_num)+".csv")
		else:
			path = os.path.join(location, "vehicle_tracks_0"+str(file_num)+".csv")
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

		if len(cleaned_trace) < 2:
			continue
		cleaned_traces.append(np.array(cleaned_trace))

	return np.array(cleaned_traces, dtype="object")

def gravity(traces, resultant_threshold, max_iter=10):
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
	repeat = True
	print("Iteration will stop when resultant is less than " + str(resultant_threshold))
	print("Processing " + str(len(points)) + " points (" + str(len(traces)) + ") traces")
	while repeat and k <= max_iter:
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
			try:
				ind = tree.query_radius(np.array([a]), r=2)
			except ValueError:
				continue
			ind[0].sort()

			# Identify all edges by finding consecutive nearby points
			# Check for incomplete edges, pair not coming from same trace?
			pairs = []
			for j in range(1, len(ind[0])):
				if ind[0][j] == ind[0][j-1] + 1:
					pairs.append((ind[0][j-1], ind[0][j]))

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

				# Compute type 1 force (gravitational)
				midpoint = (c+b)/2
				t1_distance = array_dist(a, midpoint)
				t1_direction = direction(a, midpoint)
				t1 = np.array(t1_force(t1_distance) * t1_direction)

				# Multiply t1 force by cosine of angle between edge and heading of point
				angle = theta(np.array(heading), np.array([c[0]-b[0], c[1]-b[1]]))
				t1 *= cos(angle)
				if cos(angle) < 0 and not ccw(b, c, a):
					t1 = np.array([0.0, 0.0])


				# Compute type 2 force (spring)
				if all(a == orig):
					t2 = 0
				else:
					t2_direction = direction(a, orig)
					t2 = t2_force(a, orig) * t2_direction
				
				resultant = t1 + t2

				res_mag = pow(resultant[0]**2 + resultant[1]**2, 0.5)
				if res_mag != 0:
					unit_res = resultant / res_mag
				resultants[i] += resultant

		# Move all points
		for i in range(len(points)):
			points[i] += resultants[i]

		# Recreate k-d tree with new points
		tree = KDTree(points, leaf_size=2)


		# Get max mag of resultants
		max_res = max(resultants, key=lambda v: pow(v[0]**2+ v[1]**2, 0.5))
		max_res_mag = pow(max_res[0]**2 + max_res[1]**2, 0.5)
		print("Max resultant: " + str(max_res_mag))
		if max_res_mag < resultant_threshold:
			repeat = False


	# Recreate traces using stored lengths
	new_traces = []
	for l in lengths:
		trace = []
		temp = points[:l]
		for i in range(len(temp)):
			trace.append(np.array([temp[i][0], temp[i][1]]))
		new_traces.append(np.array(trace, dtype='object'))
		points = points[l:]
	return np.array(new_traces, dtype='object')


def compute_headings(traces):
	"""
	Given a list of traces, compute the heading of each point and append.
	"""
	headings = []
	for trace in traces:
	    trace_headings = []

	    dir_vector = direction(trace[0], trace[1])
	    angle = atan2(dir_vector[1], dir_vector[0])
	    if angle < 0:
	        angle += 2*pi
	    trace_headings.append(angle)

	    for i in range(1, len(trace) - 1):
	        prev = trace[i-1]
	        next = trace[i+1]
	        dir_vector = direction(np.array([prev[0], prev[1]]), np.array([next[0], next[1]]))
	        angle = atan2(dir_vector[1], dir_vector[0])
	        if angle < 0:
	            angle += 2*pi
	        trace_headings.append(angle)

	    dir_vector = direction(trace[-2], trace[-1])
	    angle = atan2(dir_vector[1], dir_vector[0])
	    if angle < 0:
	        angle += 2*pi
	    trace_headings.append(angle)


	    headings.append(np.array([trace_headings]))

	for i in range(len(traces)):
	    traces[i] = np.concatenate((traces[i], headings[i].T), axis=1)

	return traces
