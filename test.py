from extract import dist_point_to_line
from extract import Node
import numpy as np

node1 = Node(2, 0, 0)
node2 = Node(0, 0, 0)

candidate = Node(1, 1, 0)


print(dist_point_to_line(candidate, node1, node2))





