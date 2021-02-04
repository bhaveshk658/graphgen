from graphgen.data import get_training_data, clean
from graphgen.graph import Graph, Node

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

# Get data within a certain box
box = [[960, 1015], [980, 1040]]
traces = get_training_data(1, path, box)

# Clean data and convert to nodes
trips = clean(traces, 50, 1)
for trip in trips:
    for i in range(len(trip)):
        point = trip[i]
        node = Node(point[0], point[1], point[2])
        trip[i] = node