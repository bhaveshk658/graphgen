import graphgen
from graphgen.data import get_training_data
from graphgen.graph import Graph
from graphgen.graph import Node
from graphgen.generate.generate import to_merge, convert_to_graph
from graphgen.metric import evaluate_traces
import matplotlib.pyplot as plt

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

traces = get_training_data(1, path)

trips = graphgen.data.data.clean(traces, 50, 1)

for trip in trips:
    for i in range(len(trip)):
        point = trip[i]
        node = graphgen.graph.node.Node(point[0], point[1], point[2])
        trip[i] = node


plt.xlim(969, 1015)
plt.ylim(980, 1040)

rb = [0, 1, 8, 11, 14, 15, 18, 19, 21, 24]
rb_trips = [trips[i] for i in rb]
rb_graph = convert_to_graph(rb_trips)

br = [6, 12, 13, 20, 22, 25, 26, 27]
br_trips = [trips[i] for i in br]
br_graph = convert_to_graph(br_trips)

tr = [2, 7, 10]
tr_trips = [trips[i] for i in tr]
tr_graph = convert_to_graph(tr_trips)

rt = [3, 17]
rt_trips = [trips[i] for i in rt]
rt_graph = convert_to_graph(rt_trips)

special = [4]
special_trips = [trips[i] for i in special]
special_graph = convert_to_graph(special_trips)

bt = [9, 23]
bt_trips = [trips[i] for i in bt]
bt_graph = convert_to_graph(bt_trips)

G = Graph()
G.update(rb_graph)
G.update(br_graph)
G.update(tr_graph)
G.update(rt_graph)
G.update(special_graph)
G.update(bt_graph)

G.cleanup()
G.draw()

#evaluate_traces(G, traces, 'frechet')
evaluate_traces(G, traces, 'area')
#evaluate_traces(G, traces, 'pcm')
#evaluate_traces(G, traces, 'dtw')

plt.show()

