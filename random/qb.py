import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_fnames
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from nibabel.streamlines.array_sequence import ArraySequence
import matplotlib.pyplot as plt

from test_graph import get_training_data, clean, to_merge, convert_to_graph

trips = get_training_data(1, "DR_USA_Roundabout_EP")
trips = clean(trips, 50, 1)

"""
# Remove headings
for trip in trips:
    for point in trip:
        point[2] = 0
"""
traces = ArraySequence(trips)

feature = ResampleFeature(nb_points=50)
metric = AveragePointwiseEuclideanMetric(feature=feature)
qb = QuickBundles(threshold=10., metric=metric)
clusters = qb.cluster(traces)

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", list(map(len, clusters)))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for i in range(len(clusters)):
    cluster = clusters[i]
    """
    for trip in cluster:
        for point in trip:
            color_index = i % len(colors)
            plt.scatter(point[0], point[1], c=colors[color_index])
    """
    g = convert_to_graph(cluster)
    g.draw()

plt.show()



