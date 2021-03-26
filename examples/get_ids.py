import os

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import lanelet2 as llt
from lanelet2.projection import UtmProjector

from graphgen.data import coordinate_to_id

print("Loading...")
traces = np.load(file="traces.npy", allow_pickle=True)

point = traces[0][0]
print("Point: ", (point[0], point[1]))

map_path = os.path.abspath('../data/InteractionDR1.0/maps/DR_USA_Roundabout_EP.osm')
print("Path: " + map_path)

projector = UtmProjector(llt.io.Origin(49, 8.4))
mapLoad, errors = llt.io.loadRobust(map_path, projector)

res = coordinate_to_id(mapLoad, point[0], point[1])
