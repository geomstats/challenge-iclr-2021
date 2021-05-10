from abc import ABC

import numpy as np
from geomstats.geometry.stiefel import StiefelCanonicalMetric
from .BaseStiefelMetric import BaseStiefelMetric


class StiefelFubiniStudyMetric(StiefelCanonicalMetric, BaseStiefelMetric, ABC):
	def __init__(self, n, p):
		super().__init__(n, p)

	def dist(self, point_a, point_b):
		"""Fubini-Study distance between two points.
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

		output = np.arccos(np.round(np.linalg.det(np.dot(point_a.T, point_b))))

		return output
