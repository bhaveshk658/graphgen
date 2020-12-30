from test_graph import get_training_data, clean
import similaritymeasures
import numpy as np

trips = get_training_data(1, "DR_USA_Roundabout_EP")
trips = clean(trips, 50, 1)

points0 = [None]*len(trips[0])
for i in range(len(trips[0])):
    points0[i] = trips[0][i][:2]

points1 = [None]*len(trips[1])
for i in range(len(trips[1])):
    points1[i] = trips[1][i][:2]

print(similaritymeasures.pcm(np.array(points0), np.array(points1)))