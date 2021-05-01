from abc import ABC

import numpy as np
from geomstats.geometry.stiefel import StiefelCanonicalMetric
from .BaseStiefelMetric import BaseStiefelMetric


class StiefelProjection2NormMetric(StiefelCanonicalMetric, BaseStiefelMetric, ABC):
	def __init__(self, n, p):
		super().__init__(n, p)

	def dist(self, point_a, point_b):
		"""Projection 2-Norm distance between two points.
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

		output = np.linalg.norm(np.dot(point_a, point_a.T) - np.dot(point_b, point_b.T), 2)

		return output
