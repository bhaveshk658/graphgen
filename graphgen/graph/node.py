class Node:
	'''
	Node class to use in directed graph.
	'''

	def __init__(self, x, y, heading):
		self.x = x
		self.y = y
		self.heading = heading
		self.total = 1
	
	def get_point(self):
		return [self.x, self.y, self.heading]

	def update(self, node):
		self.x = ((self.x * self.total) + node.x) / (self.total + 1)
		self.y = ((self.y * self.total) + node.y) / (self.total + 1)
		self.heading = ((self.heading * self.total) + node.heading) / (self.total + 1)
		self.total += 1
