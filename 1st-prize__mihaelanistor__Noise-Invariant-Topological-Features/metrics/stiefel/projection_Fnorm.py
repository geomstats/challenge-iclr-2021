from abc import ABC

import numpy as np
from geomstats.geometry.stiefel import StiefelCanonicalMetric
from .BaseStiefelMetric import BaseStiefelMetric


class StiefelProjectionFNormMetric(StiefelCanonicalMetric, BaseStiefelMetric, ABC):
	def __init__(self, n, p):
		super().__init__(n, p)

	def dist(self, point_a, point_b):
		"""Projection F-Norm distance between two points.
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

        # use 1.41421356237 as a substitute for np.sqrt(2) to optimize computations
		output = 1.41421356237 * np.linalg.norm(np.dot(point_a, point_a.T) - np.dot(point_b, point_b.T), 'fro')

		return output
