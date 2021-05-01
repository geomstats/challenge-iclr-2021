from abc import ABC

import numpy as np
from geomstats.geometry.stiefel import StiefelCanonicalMetric
from .BaseStiefelMetric import BaseStiefelMetric


class StiefelChordal2NormMetric(StiefelCanonicalMetric, BaseStiefelMetric, ABC):
	def __init__(self, n, p):
		super().__init__(n, p)

	def dist(self, point_a, point_b):
		"""Chordal 2-Norm distance between two points.
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

		output = np.sqrt(self.p - np.round(np.square(np.linalg.norm(np.dot(point_a.T, point_b), 2)), 4))

		return output
