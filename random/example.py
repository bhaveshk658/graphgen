import graphgen
from graphgen.data import get_training_data
from graphgen.graph import Graph
from graphgen.graph import Node
from graphgen.generate.generate import to_merge, convert_to_graph
from graphgen.metric import evaluate_traces
import matplotlib.pyplot as plt

path = "/Users/bkalisetti658/desktop/graphgen/data/InteractionDR1.0/recorded_trackfiles/DR_USA_Roundabout_EP"

box = [[960, 1015], [980, 1040]]
traces = get_training_data(1, path, box)

trips = graphgen.data.data.clean(traces, 50, 1)
for trip in trips:
    for i in range(len(trip)):
        point = trip[i]
        node = graphgen.graph.node.Node(point[0], point[1], point[2])
        trip[i] = node



plt.xlim(960, 1015)
plt.ylim(980, 1040)



rl = [16]
rl_trips = [trips[i] for i in rl]
rl_graph = convert_to_graph(rl_trips)
rl_nodes = rl_graph.get_lane_nodes()

rb = [0, 1, 8, 11, 14, 15, 18, 19, 21, 24]
rb_trips = [trips[i] for i in rb]
rb_graph = convert_to_graph(rb_trips)
rb_nodes = rb_graph.get_lane_nodes()

br = [6, 12, 13, 20, 22, 25, 26, 27]
br_trips = [trips[i] for i in br]
br_graph = convert_to_graph(br_trips)
br_nodes = br_graph.get_lane_nodes()

tr = [2, 7, 10]
tr_trips = [trips[i] for i in tr]
tr_graph = convert_to_graph(tr_trips)
tr_nodes = tr_graph.get_lane_nodes()

rt = [3]
rt_trips = [trips[i] for i in rt]
rt_graph = convert_to_graph(rt_trips)
rt_nodes = rt_graph.get_lane_nodes()

special = [4]
special_trips = [trips[i] for i in special]
special_graph = convert_to_graph(special_trips)
special_nodes = special_graph.get_lane_nodes()

bt = [9, 23]
bt_trips = [trips[i] for i in bt]
bt_graph = convert_to_graph(bt_trips)
bt_nodes = bt_graph.get_lane_nodes()



lanes = [rl_nodes, rb_nodes, br_nodes, tr_nodes, rt_nodes, special_nodes, bt_nodes]

G = convert_to_graph(lanes, dist_limit=2, heading_limit=0.2)
G.draw()

wrong = [5, 17, 28, 29]

for i in range(len(trips)):
    if i in wrong:
        continue
    for point in trips[i]:
        plt.scatter(point.x, point.y, c='r', alpha=0.2)

plt.show()
