import os
import random

import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import lanelet2 as llt
from lanelet2.core import Point3d, LineString3d, Lanelet
from lanelet2.projection import UtmProjector

from graphgen.data import coordinate_to_id, draw_map, gravity, compute_headings
from graphgen.graph import Graph, Node
from graphgen.generate import convert_to_graph, convert_to_graph_nx

print("Loading...")
map_dir = "/Users/bkalisetti658/desktop/graphgen/data/DR_USA_Roundabout_EP/"

map_path = os.path.abspath('data/InteractionDR1.0/maps/DR_USA_Roundabout_EP.osm')

projector = UtmProjector(llt.io.Origin(0, 0))
mapLoad, errors = llt.io.loadRobust(map_path, projector)

with open("graph.x", 'rb') as f:
    G = pickle.load(f)

i = 1
for edge in G.edges:
    p1 = edge[0]
    p2 = edge[1]

    id1 = random.randint(5000, 1000000)

    linestring = LineString3d(-i, [Point3d(id1, G.nodes[p1]['x'], G.nodes[p1]['y'], 0), Point3d(id1+1, G.nodes[p2]['x'], G.nodes[p2]['y'], 0)])
    lanelet = Lanelet(i, linestring, linestring)
    mapLoad.add(lanelet)
    i += 1
llt.io.write("new.osm", mapLoad, projector)
