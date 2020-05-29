from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric


class FrechetDistance(Metric):
	'''
	Computes Frechet Distance between two trajectories.
	'''
	def __init__(self):
		super(FrechetDistance, self).__init__(feature=ResampleFeature(nb_points=256))

	def are_compatible(self, shape1, shape2):
		return len(shape1) == len(shape2)

	def dist(self, v1, v2):
		return frdist(v1, v2)


counter = 0
merge_counter = 0
other_counter = 0
for trip in trips:
	prevNode = None
	for n in trip:
		counter += 1
		(merge, closest_edge, closest_node, short_projection_distance) = to_merge(n, G)
		if merge:
			merge_counter += 1
			if short_projection_distance > 1:
				if node_dist(n,closest_edge[0]) > 15 or node_dist(n,closest_edge[1]) > 15:
					continue
				G.add_edge(closest_edge[0], n)
				G.add_edge(n, closest_edge[1])
				G.remove_edge(closest_edge[0], closest_edge[1])
				#check = True
			else:
				closest_node.merge(n)
				n = closest_node
				#check = False
				print(nx.has_path(G, prevNode, n))
				if prevNode is not None and nx.has_path(G, prevNode, n) and len(nx.shortest_path(G, prevNode, n)) > 2:
					print("hi")
					G.add_edge(prevNode, n)
					prevNode = n

		else:
			other_counter += 1
			if prevNode is not None and node_dist(n, prevNode) > 15:
				continue
			G.add_node(n)
			if prevNode is not None:
				G.add_edge(prevNode, n)
			prevNode = n




    data = pd.read_csv(os.path.join("interaction-dataset-copy/recorded_trackfiles/DR_USA_Roundabout_EP", "vehicle_tracks_000.csv"))
    box = [[975, 1000], [985, 1010]]
    box2 = [[1055, 1085], [1005, 1025]]

    ### Plot all data points in the csv ###

    total_data = data.to_numpy()
    total_data = np.vstack((total_data[:, 4], total_data[:, 5])).T
    plt.figure(1)
    #plt.scatter(total_data[:, 0], total_data[:, 1], s=1)
    plt.title("All data points + K-means clusters")
    plt.xlabel("X-coordinates")
    plt.ylabel("Y-coordinates")


    ### K-means clustering ###

    Kmean = KMeans(n_clusters=250)
    Kmean.fit(total_data)
    centroids = Kmean.cluster_centers_
    for i in range(len(centroids)):
    	plt.scatter(centroids[i][0], centroids[i][1], s=10)

    ###Cars of interest - 16, 23 ###

    d16 = data.loc[(data['track_id'] == 16) & (data['x'] > box[0][0]) & (data['x'] < box[0][1])
    	& (data['y'] > box[1][0]) & (data['y'] < box[1][1])].to_numpy()

    d16 = np.vstack((d16[:, 4], d16[:, 5], d16[:, 2])).T

    d23 = data.loc[(data['track_id'] == 23) & (data['x'] > box[0][0]) & (data['x'] < box[0][1])
    	& (data['y'] > box[1][0]) & (data['y'] < box[1][1])].to_numpy()
    d23 = np.vstack((d23[:, 4], d23[:, 5], d23[:, 2])).T


    ### Predict cluster of each data point in box ###
    '''
    Find merge point by iterating through centroids
    Fit curve to car 16's clusters and fit another to car 23's clusters
    	Represent functions for lanes/the road
    Use actual x-coordinates in d16/d23 to determine locations on the road using functions
    	Get arc length from that point to merge point (either is already on function or recalculate)
    	Map timestamp to arc length

    Plot arc length (distance) vs time



    '''
    interest16 = Kmean.predict(d16[:, :-1])
    interest23 = Kmean.predict(d23[:, :-1])

    ### Find merge point and coordinates of the clusters that the cars travel through###

    cluster_coord_16 = []
    cluster_coord_23 = []

    for i in range(len(centroids)):
    	if i in interest16:
    		cluster_coord_16.append(centroids[i])
    	if i in interest23:
    		cluster_coord_23.append(centroids[i])

    cluster_coord_16 = np.array(cluster_coord_16)
    cluster_coord_23 = np.array(cluster_coord_23)

    poly_16 = np.polyfit(cluster_coord_16[:, 0], cluster_coord_16[:, 1], 3)
    poly_23 = np.polyfit(cluster_coord_23[:, 0], cluster_coord_23[:, 1], 4)

    d16_pred = np.vstack((d16[:, 0], np.polyval(poly_16, d16[:, 0]), d16[:, 2])).T
    plt.plot(d16_pred[:, 0], d16_pred[:, 1], c='b')

    d23_pred = np.vstack((d23[:, 0], np.polyval(poly_23, d23[:, 0]))).T
    plt.plot(d23_pred[:, 0], d23_pred[:, 1], c='r')

    temp_x = np.linspace(975, 1100, 200)
    temp_y1 = np.polyval(poly_16, temp_x)
    temp_y2 = np.polyval(poly_23, temp_x)

    idx = np.argwhere(np.diff(np.sign(temp_y2 - temp_y1))).flatten()

    merge_point = [temp_x[idx], temp_y1[idx]]

    plt.scatter(cluster_coord_16[:, 0], cluster_coord_16[:, 1], c='b')
    plt.scatter(cluster_coord_23[:, 0], cluster_coord_23[:, 1], c='r')
    plt.scatter(merge_point[0], merge_point[1], c='g')

    ### Isolate points with proper timestamps ###
    d23_time = []

    for i in range(len(d23)):
    	if d23[i][2] in d16[:, 2]:
    		d23_time.append(d23[i])

    d23 = np.array(d23_time)

    d23_pred = np.vstack((d23[:, 0], np.polyval(poly_23, d23[:, 0]), d23[:, 2])).T


    ### Redefine as functions ###
    def f(x):
    	return np.polyval(poly_16, x)

    def g(x):
    	return np.polyval(poly_23, x)

    dist_16 = []
    dist_23 = []
    ### Compute arc lengths ###
    for i in d16_pred:
    	dist_16.append([i[2], arclength(f, i[0], merge_point[0])])
    for j in d23_pred:
    	dist_23.append([j[2], arclength(g, j[0], merge_point[0])])

    dist_16 = np.array(dist_16)
    dist_23 = np.array(dist_23)



    plt.figure(2)
    plt.title("Distance to merge point versus time")
    plt.xlabel("Time (ms)")
    plt.ylabel("Distance to point on path closest to merge point")
    plt.plot(dist_16[:, 0], dist_16[:, 1], c='b')
    plt.plot(dist_23[:, 0], dist_23[:, 1], c='r')

    plt.show()



    temp = []
	for i in traces:
		if len(i) >= 50:
			temp.append(i)
	clusters = quick_bundles(temp)
	paths = []
	for i in range(len(clusters)):
		if len(clusters[i].indices) < 6:
			continue
		path = []
		for i in clusters[i].indices:
			path.append(temp[i])
		paths.append(path)
	for i in range(len(paths)):
		points = np.array([item for sublist in paths[i] for item in sublist])
		plot_clusters(points)