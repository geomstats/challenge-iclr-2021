"""Riemannian mean-shift clustering."""
import numpy as np
import joblib

from sklearn.base import BaseEstimator, ClusterMixin

from geomstats.learning.frechet_mean import FrechetMean

class RiemannianMeanShift(ClusterMixin, BaseEstimator):

    def __init__(
            self, manifold, metric, bandwidth, tol=1e-2, **FrechetMean_kwargs):
        
        self.manifold = manifold
        self.metric = metric
        self.bandwidth = bandwidth
        self.tol = tol
        self.mean = FrechetMean(self.metric, **FrechetMean_kwargs)
        self.centers = None
        
    # parallel computation of distances between two sets of points, without intraset distances
    # as in RiemannianMetric.dist_pairwise
    def __intersets_distances(self, points_A, points_B, n_jobs=1, **joblib_kwargs):
        """
        parallel computation of distances between two sets of points, without intraset distances
        as in RiemannianMetric.dist_pairwise
        """
        n_A, n_B = points_A.shape[0], points_B.shape[0]
        
        
        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_dist(x, y):
            return self.metric.dist(x, y)
        
        pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
        out = pool(pickable_dist(points_A[i,:], points_B[j,:]) for i in range(n_A) for j in range(n_B))
        
        # assuming numpy backend
        return np.array(out).reshape((n_A,n_B))
    
    
    def fit(self, points, n_centers, n_jobs = 1, max_iter = 100, 
            init_centers = 'from_points', kernel = 'flat', **joblib_kwargs):
        
        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_mean(points, weights):
            return self.mean.fit(points, weights = weights).estimate_
        
        # init from points, since uniform is only for compact manifolds
        # from_points should be default
        if init_centers == 'from_points':
            n_points = points.shape[0]
            centers = points[np.random.randint(n_points,size=n_centers),:]
        if init_centers == 'random_uniform':
            centers = self.manifold.random_uniform(n_samples = n_centers)
            n_centers = centers.shape[0]
        
        for i in range(max_iter):
            dists = self.__intersets_distances(centers, points, n_jobs=n_jobs, **joblib_kwargs)
            
            if kernel == 'flat':
                weights = np.ones_like(dists)
            
            weights[dists > self.bandwidth] = 0.
            weights = weights/weights.sum(axis=1,keepdims=1)              
                        
            points_to_average, nonzero_weights = [], []
            
            for j in range(n_centers):
                points_to_average += [points[np.where(weights[j,:] > 0)[0],:], ]
                nonzero_weights += [weights[j,:].nonzero()[0], ]
                
            # compute Frechet means in parallel

            pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
            out = pool(pickable_mean(points_to_average[j], nonzero_weights[j]) for j in range(centers.shape[0]))
            
            new_centers = np.array(out)

            displacements = [self.metric.dist(centers[j], new_centers[j]) for j in range(n_centers)]

            centers = new_centers
            
            if (np.array(displacements) < self.tol).all():
                break
            
        self.centers = centers
        
        
    def predict(self, X):
        """
        Predict the closest cluster each point in X belongs to
        """
            
        if self.centers is None:
            raise Exception("Not fitted")
        else:
            out = []
            for i in range(X.shape[0]):
                j_closest_center = self.metric.closest_neighbor_index(X[i,:], self.centers)
                out.append(self.centers[j_closest_center, :])
                
            return np.array(out)
        
    
        
        
                
                
                
            
            
            
            
            
        
        
        
        
        
        
        
        
        
    