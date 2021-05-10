import numpy as np


class BaseStiefelMetric:
    
	# point_a, point_b are flatten
	def array_dist(self, point_a, point_b):
		point_a = np.reshape(point_a, (self.n, self.p))
		point_b = np.reshape(point_b, (self.n, self.p))
		return self.dist(point_a, point_b)

	# lists of points flatten
	def views_dist(self, points_a, points_b):
		assert points_a.shape[0] == points_b.shape[0]
        
		views = int(points_a.shape[0] / (self.n * self.p))

		points_a = np.reshape(points_a, (views, self.n, self.p))
		points_b = np.reshape(points_b, (views, self.n, self.p))

		dist = 0.0
		for i in range(points_a.shape[0]):
			dist += self.dist(points_a[i], points_b[i])

		return dist