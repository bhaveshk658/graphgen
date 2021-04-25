import numpy as np
import matplotlib.pyplot as plt
#import lanelet2 as llt
#from lanelet2.core import BasicPoint2d

from graphgen.data.llt_data_utils import *

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


def draw_map(path):
    """
    Draws physical map of .osm file provided by path.
    """
    fig, axes = plt.subplots(1, 1)
    fig.canvas.set_window_title("Visualization")

    lat_origin = 0.
    long_origin = 0.

    projector = llt.projection.UtmProjector(llt.io.Origin(lat_origin, long_origin))
    laneletmap = llt.io.load(path, projector)
    draw_lanelet_map(laneletmap, axes)
