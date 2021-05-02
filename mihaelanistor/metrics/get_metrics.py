from geomstats.geometry.stiefel import Stiefel, StiefelCanonicalMetric
from geomstats.geometry.grassmannian import Grassmannian, GrassmannianCanonicalMetric

from metrics.stiefel.subspace_angle import StiefelSubspaceAngleMetric
from metrics.stiefel.chordal_distance_Fnorm import StiefelChordalFNormMetric
from metrics.stiefel.chordal_distance_2norm import StiefelChordal2NormMetric
from metrics.stiefel.fubini_study import StiefelFubiniStudyMetric
from metrics.stiefel.projection_2norm import StiefelProjection2NormMetric
from metrics.stiefel.projection_Fnorm import StiefelProjectionFNormMetric

from metrics.grassmannian.subspace_angle import GrassmannianSubspaceAngleMetric
from metrics.grassmannian.chordal_distance_Fnorm import GrassmannianChordalFNormMetric
from metrics.grassmannian.chordal_distance_2norm import GrassmannianChordal2NormMetric
from metrics.grassmannian.fubini_study import GrassmannianFubiniStudyMetric
from metrics.grassmannian.projection_2norm import GrassmannianProjection2NormMetric
from metrics.grassmannian.projection_Fnorm import GrassmannianProjectionFNormMetric



def get_stiefel_metrics(stiefel, point_a, point_b):
	assert isinstance(stiefel, Stiefel)
	assert stiefel.belongs(point_a)
	assert stiefel.belongs(point_b)
	results = {}

	metric = StiefelChordalFNormMetric(stiefel.n, stiefel.p)
	results["chordal_distance"] = metric.dist(point_a, point_b)

	metric = StiefelSubspaceAngleMetric(stiefel.n, stiefel.p)
	results["subspace_angle"] = metric.dist(point_a, point_b)

	metric = StiefelCanonicalMetric(stiefel.n, stiefel.p)
	results["canonical_distance"] = metric.dist(point_a, point_b)

	return results


def get_grassmannian_metrics(grassmannian, point_a, point_b):
	assert isinstance(grassmannian, Grassmannian)
	assert grassmannian.belongs(point_a)
	assert grassmannian.belongs(point_b)
	results = {}

	metric = GrassmannianChordalFNormMetric(grassmannian.n, grassmannian.k)
	results["chordal_distance"] = metric.dist(point_a, point_b)

	metric = GrassmannianSubspaceAngleMetric(grassmannian.n, grassmannian.k)
	results["subspace_angle"] = metric.dist(point_a, point_b)

	metric = GrassmannianCanonicalMetric(grassmannian.n, grassmannian.k)
	results["canonical_distance"] = metric.dist(point_a, point_b)

	return results

def get_callable_stiefel_metrics(stiefel):
	assert isinstance(stiefel, Stiefel)
	metrics = {}

	metric = StiefelSubspaceAngleMetric(stiefel.n, stiefel.p)
	metrics["subspace_angle"] = metric.dist

	metric = StiefelChordalFNormMetric(stiefel.n, stiefel.p)
	metrics["chordal_distance_Fnorm"] = metric.dist

	metric = StiefelChordal2NormMetric(stiefel.n, stiefel.p)
	metrics["chordal_distance_2norm"] = metric.dist

	metric = StiefelFubiniStudyMetric(stiefel.n, stiefel.p)
	metrics["fubini_study"] = metric.dist

	metric = StiefelProjection2NormMetric(stiefel.n, stiefel.p)
	metrics["projection_2norm"] = metric.dist

	metric = StiefelProjectionFNormMetric(stiefel.n, stiefel.p)
	metrics["projection_Fnorm"] = metric.dist

	# Todo: uncomment
	# metric = StiefelCanonicalMetric(stiefel.n, stiefel.p)
	# metrics["canonical_distance"] = metric.dist

	return metrics

def get_callable_stiefel_metrics_array(stiefel):
	assert isinstance(stiefel, Stiefel)
	metrics = {}

	metric = StiefelSubspaceAngleMetric(stiefel.n, stiefel.p)
	metrics["subspace_angle"] = metric.array_dist

	metric = StiefelChordalFNormMetric(stiefel.n, stiefel.p)
	metrics["chordal_distance_Fnorm"] = metric.array_dist

	metric = StiefelChordal2NormMetric(stiefel.n, stiefel.p)
	metrics["chordal_distance_2norm"] = metric.array_dist

	metric = StiefelFubiniStudyMetric(stiefel.n, stiefel.p)
	metrics["fubini_study"] = metric.array_dist

	metric = StiefelProjection2NormMetric(stiefel.n, stiefel.p)
	metrics["projection_2norm"] = metric.array_dist

	metric = StiefelProjectionFNormMetric(stiefel.n, stiefel.p)
	metrics["projection_Fnorm"] = metric.array_dist

	# Todo: uncomment
	# metric = StiefelCanonicalMetric(stiefel.n, stiefel.p)
	# metrics["canonical_distance"] = metric.dist

	return metrics

def get_callable_stiefel_metrics_views(stiefel):
	assert isinstance(stiefel, Stiefel)
	metrics = {}

	metric = StiefelSubspaceAngleMetric(stiefel.n, stiefel.p)
	metrics["subspace_angle"] = metric.views_dist

	metric = StiefelChordalFNormMetric(stiefel.n, stiefel.p)
	metrics["chordal_distance_Fnorm"] = metric.views_dist

	metric = StiefelChordal2NormMetric(stiefel.n, stiefel.p)
	metrics["chordal_distance_2norm"] = metric.views_dist

	metric = StiefelFubiniStudyMetric(stiefel.n, stiefel.p)
	metrics["fubini_study"] = metric.views_dist

	metric = StiefelProjection2NormMetric(stiefel.n, stiefel.p)
	metrics["projection_2norm"] = metric.views_dist

	metric = StiefelProjectionFNormMetric(stiefel.n, stiefel.p)
	metrics["projection_Fnorm"] = metric.views_dist

	# Todo: uncomment
	# metric = StiefelCanonicalMetric(stiefel.n, stiefel.p)
	# metrics["canonical_distance"] = metric.dist

	return metrics

def get_callable_grassmannian_metrics(grassmannian):
	assert isinstance(grassmannian, Grassmannian)
	metrics = {}

	metric = GrassmannianSubspaceAngleMetric(grassmannian.n, grassmannian.p)
	metrics["subspace_angle"] = metric.dist

	metric = GrassmannianChordalFNormMetric(grassmannian.n, grassmannian.p)
	metrics["chordal_distance_Fnorm"] = metric.dist

	metric = GrassmannianChordal2NormMetric(grassmannian.n, grassmannian.p)
	metrics["chordal_distance_2norm"] = metric.dist

	metric = GrassmannianFubiniStudyMetric(grassmannian.n, grassmannian.p)
	metrics["fubini_study"] = metric.dist

	metric = GrassmannianProjection2NormMetric(grassmannian.n, grassmannian.p)
	metrics["projection_2norm"] = metric.dist

	metric = GrassmannianProjectionFNormMetric(grassmannian.n, grassmannian.p)
	metrics["projection_Fnorm"] = metric.dist

	# Todo: uncomment
	# metric = GrassmannianCanonicalMetric(grassmannian.n, grassmannian.k)
	# metrics["canonical_distance"] = metric.dist

	return metrics
