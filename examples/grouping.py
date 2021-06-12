from graphgen.generate.generate import convert_to_graph_nx
import os
import numpy as np
import pickle

import lanelet2 as llt
from lanelet2.core import BasicPoint2d
from lanelet2.projection import UtmProjector

def coordinate_to_id(llt_map, x, y):
	"""
	    Given a map filename and a local x and y coordinate pair, returns a list of the IDs of the lanelets that are
	    closest to the given coordinate. If the coordinate intersects with multiple lanelets, returns IDs of all
	    intersecting lanelets.
	"""
	all_lanelets = []
	point = np.array([[x], [y]])
	point = BasicPoint2d(point[0][0], point[1][0])
	lanelets = llt.geometry.findWithin2d(llt_map.laneletLayer, point)              # gets lanelets point lies in
	all_lanelets.extend([lanelet[1].id for lanelet in lanelets])

	id_dict = {llt.id: index for (index, llt) in enumerate(llt_map.laneletLayer)}
	all_lanelets = [id_dict[int(x)] for x in np.unique(all_lanelets)]               # gets all unique lanelets
	return all_lanelets

traces = np.load(file="traces.npy", allow_pickle=True)


map_path = os.path.abspath('data/InteractionDR1.0/maps/DR_USA_Roundabout_EP.osm')

projector = UtmProjector(llt.io.Origin(0, 0))
mapLoad, errors = llt.io.loadRobust(map_path, projector)

d = dict()
ids = []
for i in range(len(traces)):
    trace = traces[i]
    trace_id = []
    for point in [trace[0], trace[-1]]:
        res = coordinate_to_id(mapLoad, point[0], point[1])
        if len(res) == 0:
            res = float('inf')
        else:
            res.sort()
            res = res[0]
        trace_id.append(res)
    trace_id = tuple(trace_id)
    if trace_id in d:
        d[trace_id].append(i)
    else:
        d[trace_id] = [i]

with open('groups.x', 'wb') as f:
    pickle.dump(d, f)