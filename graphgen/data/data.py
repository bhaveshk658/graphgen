import os
from pandas import read_csv
import numpy as np
from scipy.interpolate import interp1d
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import time
import random

from graphgen.data.utils import *
from graphgen.graph import Node

def get_training_data(n, location, box=None):
	"""
	Get training data from n files from string location.
	E.g. get_training_data(4, "path/to/data")
	Returns a list of all traces
	"""
	traces = []

	for i in range(n):
		path = os.path.join(location, "vehicle_tracks_00"+str(i)+".csv")
		data = read_csv(path)

		# Define rectangle of area to take points from.
		if box:
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
	for _ in range(1):

		resultants = [0]*len(points)
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
			
			if (i == rand_index):
				plt.scatter(a[0], a[1], c='b') # Point
				plt.scatter(points[i-1][0], points[i-1][1], c='b') # Point before
				plt.scatter(points[i+1][0], points[i+1][1], c='b') # Point after
				plt.scatter(a[0] + d[0], a[1] + d[1], c='k') # Heading
				plt.scatter(a[0]-d[0], a[1]-d[1], c='k') # Heading
			

			# Iterate through all edges, identifying intersecting ones
			intersecting_pairs = []
			for pair in pairs:
				b = points[pair[0]]
				c = points[pair[1]]
				midpoint = (c+b)/2
				# If edge intersects perpendicular line from point, proceed
				if line_line_segment_intersect(a, d, b, c):
					intersecting_pairs.append(pair)

			# Iterate through intersecting edges
			for pair in intersecting_pairs:
				b = points[pair[0]]
				c = points[pair[1]]
				
				if i == rand_index:
					plt.scatter(b[0], b[1], c='g', alpha=0.4)
					plt.scatter(c[0], c[1], c='g', alpha=0.4)
				
				midpoint = (c+b)/2
				t1_distance = dist(a, midpoint)
				t1_direction = direction(a, midpoint)
				t1 = np.array(t1_force(t1_distance) * t1_direction)


				if all(a == orig):
					t2 = 0
				else:
					t2_direction = direction(a, orig)
					t2 = t2_force(a, orig) * t2_direction

				resultant = t1 + t2
				if i == rand_index:
					a += (resultant)#3.7 is the number
				resultants[i] += pow(resultant[0]**2 + resultant[1]**2, 0.5)

			if i == rand_index:
				plt.scatter(a[0], a[1], c='y')
			
			points[i] = a
			# Update kd tree
			tree = KDTree(points, leaf_size=2)
		print(len(resultants), len(points))
		print("Max resultant: "+str(max(resultants)))
		print("Min resultant: "+str(min(resultants)))
		print("Avg resultant: "+str(sum(resultants)/len(resultants)))


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
		

		