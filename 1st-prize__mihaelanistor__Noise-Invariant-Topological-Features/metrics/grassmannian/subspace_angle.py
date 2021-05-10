from abc import ABC

import numpy as np
from geomstats.geometry.grassmannian import GrassmannianCanonicalMetric


class GrassmannianSubspaceAngleMetric(GrassmannianCanonicalMetric, ABC):
	def __init__(self, n, p):
		super().__init__(n, p)

	def dist(self, point_a, point_b):
		"""Subspace Angle distance between two points.

		Parameters
		----------
		point_a : array-like, shape=[..., dim]
			Point.
		point_b : array-like, shape=[..., dim]
			Point.
		Returns
		-------
		dist : array-like, shape=[...,]
			Distance.
		"""

		temp = np.dot(point_a.T, point_b)
		U, S, V = np.linalg.svd(temp)
		S = np.round(np.array(S), 4)
		S = np.sort(S)[::-1][0:self.p]
		thetas = np.arccos(S)

		output = np.sqrt(np.sum(np.square(thetas)))

		return output
