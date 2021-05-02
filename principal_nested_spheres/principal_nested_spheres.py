import numpy as np
import scipy
from scipy import optimize as opt
from sklearn.decomposition import PCA
from utils import *
from functools import partial
class PNS(object):
    """
    Fit nested_spheres to data. This is a python code to PNS matlab code
    See Sungkyu Jung et al, 2012 for the original PNS.
    For Kurtosis test, see Byungwon Kim et al., 2020 for reference.
    For an application on shape analysis, refer to Liu et al., Non-Euclidean Analysis of Joint Variations in Multi-object Shapes.

    There might be some small differences than the matlab implementation due to the different optimization methods and other numerical issues (e.g., non-uniqueness of singular vectors from SVD).

    Author: Zhiyuan Liu
    Data: Oct. 10, 2020
    """

    def __init__(self, data=None, itype=9, alpha=0.05):
        """
        Initialize an object of PNS model for data with fitting type itype.
        
        Args:
            data (numpy.ndarray): A 2D matrix of dimension d-by-N, where d is the number of features and N is the number of  cases
            itype (integer): The type of the fitting strategy
                ################### test type ######################
                ## itype = 1: always small circle
                ## itype = 2: always great circle
                ## itype = 9 (default): apply Kurtosis test to automatically choose between great and small circle fitting
            alpha (float): significance level for testing of whether great or small circle

        Attributes:
            output (tuple): The result of PNS fitting, including
                resmat (numpy.ndarray): The Euclideanized features in a matrix of dimension (k-1)-by-k, where k = min(d, N)
                PNS (tuple): The configuration of the fitted coordinate system, which is composed of
                        0. radii (list): denote the size of each fitted subsphere for the use of normalize residuals
                        1. orthaxis (list): one of the parameters of every fitted subsphere. Centers of fitted subspheres
                        2. dist (list): another parameter (geodesic distance in radians) of every fitting subsphere
                        3. pvalues (list): intermediate results from hypothesis testing for every subsphere fitting. It's empty if itype != 9
                        4. gsphere (list): types (great sphere:1 or small sphere:0) of fitted subspheres
                        5. basisu (list): the loadings in the embedding feature space
                        6. mean (vector): PNS mean. This is the center of the distribution
                        7. itype (list): user-selected types (great sphere:2 or small sphere:1)
        Returns:
            An instance of a PNS model

        """
        ## Input: d x n matrix, where d is the number of features
        self.data = data
        self.itype = itype
        ## significance level for testing of whether great or small circle
        self.alpha = alpha

        ## output: (resmat, PNS)
        self.output = None
    def emsvd(self, Y, k=None, tol=1E-3, maxiter=None):
        """
        Approximate SVD on data with missing values via expectation-maximization

        Inputs:
        -----------
        Y:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf
        k:          number of singular values/vectors to find (default: k=ndim)
        tol:        convergence tolerance on change in trace norm
        maxiter:    maximum number of EM steps to perform (default: no limit)

        Returns:
        -----------
        Y_hat:      (nobs, ndim) reconstructed data matrix
        mu_hat:     (ndim,) estimated column means for reconstructed data
        U, s, Vt:   singular values and vectors (see np.linalg.svd and 
                    scipy.sparse.linalg.svds for details)
        """

        if k is None:
            svdmethod = partial(np.linalg.svd, full_matrices=False)
        else:
            svdmethod = partial(svds, k=k)
        if maxiter is None:
            maxiter = np.inf

        # initialize the missing values to their respective column means
        mu_hat = np.nanmean(Y, axis=0, keepdims=1)
        valid = np.isfinite(Y)
        Y_hat = np.where(valid, Y, mu_hat)

        halt = False
        ii = 1
        v_prev = 0

        while not halt:

            # SVD on filled-in data
            U, s, Vt = svdmethod(Y_hat - mu_hat)

            # impute missing values
            Y_hat[~valid] = (U.dot(np.diag(s)).dot(Vt) + mu_hat)[~valid]

            # update bias parameter
            mu_hat = Y_hat.mean(axis=0, keepdims=1)

            # test convergence using relative change in trace norm
            v = s.sum()
            if ii >= maxiter or ((v - v_prev) / v_prev) < tol:
                halt = True
            ii += 1
            v_prev = v

        return Y_hat, mu_hat, U, s, Vt
    def fit(self):
        """
        This is the main entry of fitting PNS to data

        """
        ## 0. make sure the data are distributed on a unit sphere
        d, n = self.data.shape
        if not is_on_unit_sphere(self.data):
            print("Mapping data to preshape space")
            data_in_3d = np.reshape(self.data, (-1, 3, n))
            _, k_landmarks, _ = data_in_3d.shape
            from geomstats.geometry.pre_shape import PreShapeSpace

            preshape = PreShapeSpace(m_ambient=3, k_landmarks=k_landmarks)
            data_preshape = preshape.projection(data_in_3d)
            base_point = data_preshape[0]

            data_shape = preshape.align(point=data_preshape, base_point=base_point)
            
            self.data = np.reshape(data_shape, (d, n))
        ## 1. rotate data to get a tight space, excluding the null space
        eps = 1e-15
        
        u, s, _ = np.linalg.svd(self.data, full_matrices=False)
        small_singular_val = np.where(s < eps)[0]
        maxd = len(small_singular_val)
        if maxd == 0:
            maxd = np.min([d, n]) + 1

        ## the dimension of null space
        nullspdim = d - maxd + 1

        ## 2. intrinsic dimension of sphere is 1 dimension lower than extrinsic_dim
        dm = maxd - 2

        basisu = []
        if nullspdim > 0:
            basisu = u[:, :dm+1]
            ## extract the signal by projecting to the kernel space (complementary of the null space)
            currentSphere = np.matmul(u[:, :dm+1].T, self.data)

        else:
            currentSphere = self.data

        if self.itype == 9:
            ## Use hypothesis testing (Kurtosis test) to decide whether great or small circle for EACH subsphere
            self.output = self.automatic_fit_subspheres(currentSphere, dm, nullspdim, basisu)
        else:
            ## Otherwise, always fit data with one particular circle type (great or small)
            self.output = self.fit_with_subspheres(currentSphere, dm, nullspdim, basisu)
    def automatic_fit_subspheres(self, data, dm, nullspdim, basisu=[]):
        """
        Automatically decide which type (great or small) spheres to fit the data

        Args:
            data (numpy.ndarray): A 2D matrix of dimension d-by-N, where d is the number of features and N is the number of  cases
            dm (integer): the intrinsic dimension of the hypersphere
            nullspdim (integer): the dimension of the null space
            basisu (list): the input basis

        Returns:
            resmat (numpy.ndarray): The Euclideanized features in a matrix of dimension (k-1)-by-k, where k = min(d, N)
            PNS (tuple): The configuration of the fitted coordinate system, which is composed of
                        0. radii (list): denote the size of each fitted subsphere for the use of normalize residuals
                        1. orthaxis (list): one of the parameters of every fitted subsphere. Centers of fitted subspheres
                        2. dist (list): another parameter (geodesic distance in radians) of every fitting subsphere
                        3. pvalues (list): intermediate results from hypothesis testing for every subsphere fitting. It's empty if itype != 9
                        4. gsphere (list): types (great sphere:1 or small sphere:0) of fitted subspheres
                        5. basisu (list): the loadings in the embedding feature space
                        6. mean (vector): PNS mean. This is the center of the distribution
                        7. itype (list): user-selected types (great sphere:2 or small sphere:1)
        """
        def LRTpval(res_great, res_small, n):
            chi2 = n * np.log(np.sum(res_great ** 2) / np.sum(res_small ** 2))
            chi2 = max(chi2, 0)
            return 1 - scipy.stats.chi2.cdf(chi2, 1)
        def decide_circle_type(dim, small_circle=True):
            circle_type = 'SMALL' if small_circle else 'GREAT'
            print(str(dim) + '-sphere to ' + str(dim-1) + '-sphere by fitting a  '+ circle_type +'  sphere')
        dist = []
        resmat = []
        orthaxis = []
        gsphere = []
        pvalues = []
        iso = []
        _, num_cases = data.shape
        nan = float('nan')
        print('Testing with kurtosis using alpha: ' + str(self.alpha))
        is_isotropic = False
        for i in range(dm - 1):
            center, r = None, None
            if is_isotropic:
                decide_circle_type(dm-i, False)

                center, r = self.get_subsphere(data)
                gsphere.append(1)
                pvalues.append((nan, nan))
            else:

                center_small, r_small = self.get_subsphere(data, True)
                small_rot_data = np.matmul(center_small.T, data)
                res_small = np.arccos(np.clip(small_rot_data, -1, 1)) - r_small

                center_great, r_great = self.get_subsphere(data)
                great_rot_data = np.matmul(center_great.T, data)
                res_great = np.arccos(np.clip(great_rot_data, -1, 1)) - r_great

                ## Chi-squared statistic for a likelihood test
                pval1 = LRTpval(res_great, res_small, num_cases)
                if pval1 > self.alpha:
                    center, r = center_great, r_great
                    pvalues.append((pval1, nan))
                    gsphere.append(1)
                    decide_circle_type(dm-i, False)
                else:
                    ## Kurtosis test
                    data_centered_around_np = rotate_to_north_pole(center_small.squeeze()) @ data
                    data_in_tangent = log_north_pole(data_centered_around_np)
                    d, n = data_in_tangent.shape
                    norm_data = np.sum(data_in_tangent ** 2, axis=0)
                    kurtosis = np.sum(norm_data ** 2) / float(n) / (np.sum(norm_data) / (d*(n-1))) ** 2
                    M_kurt = d * (d + 2) ** 2 / (d + 4)
                    V_kurt = (1/n) * (128*d*(d+2)^4) / ((d+4)^3*(d+6)*(d+8))
                    pval2 = scipy.stats.norm.cdf((kurtosis - M_kurt) / np.sqrt(V_kurt))

                    pvalues.append((pval1, pval2))
                    if pval2 > self.alpha:
                        center, r = center_great, r_great
                        gsphere.append(1)
                        decide_circle_type(dm - i, False)
                        is_isotropic = True
                    else:
                        center, r = center_small, r_small
                        gsphere.append(0)
                        decide_circle_type(dm - i)
            res_angle = np.matmul(center.T, data)
            res = np.arccos(np.clip(res_angle, -1, 1)) - r
            orthaxis.append(center)

            dist.append(r)
            resmat.append(res.squeeze())
            iso.append(is_isotropic)

            nested_sphere = np.matmul(rotate_to_north_pole(center.squeeze()), data)
            data = nested_sphere[:dm-i, :] / np.sqrt(1-nested_sphere[dm-i, :] ** 2)[np.newaxis,:]

        ## parameterize 1-sphere to angles
        if True: #nullspdim + 1 - (dm - 1) <= 0:
            s1_to_radian = np.arctan2(data[1, :], data[0, :])
            mean_theta, _ = self.geod_mean_s1(s1_to_radian.T)
            orthaxis.append(mean_theta)
            last_res = (s1_to_radian - mean_theta + np.pi) % (2*np.pi) - np.pi
            resmat.append(last_res)
        ## scale resmat according to the sizes of subspheres
        radii = [1.0]
        for i in range(1, dm):
            radii.append(np.prod(np.sin(dist[:i])))
        resmat = np.flipud(np.array(radii)[:, np.newaxis] * resmat)

        PNS = {'radii': radii, 'orthaxis': orthaxis, 'dist': dist, 'pvalues': pvalues, \
                'gsphere': gsphere, 'basisu': basisu, 'mean': [], 'itype': self.itype}
        PNS['mean'] = self.inv(np.zeros((dm, 1)), PNS)
        return (resmat, PNS)

    def fit_with_subspheres(self, data, dm, nullspdim, basisu=[]):
        """
        Fit the data with user-selected types (great or small sphere) of subspheres

        Args:
            data (numpy.ndarray): A 2D matrix of dimension d-by-N, where d is the number of features and N is the number of  cases
            dm (integer): the intrinsic dimension of the hypersphere
            nullspdim (integer): the dimension of the null space
            basisu (list): the input basis

        Returns:
            resmat (numpy.ndarray): The Euclideanized features in a matrix of dimension (k-1)-by-k, where k = min(d, N)
            PNS (tuple): The configuration of the fitted coordinate system, which is composed of
                        0. radii (list): denote the size of each fitted subsphere for the use of normalize residuals
                        1. orthaxis (list): one of the parameters of every fitted subsphere. Centers of subspheres.
                        2. dist (list): another parameter (geodesic distance in radians) of every fitting subsphere
                        3. pvalues (list): intermediate results from hypothesis testing for every subsphere fitting. It's empty if itype != 9
                        4. gsphere (list): types (great sphere:1 or small sphere:0) of fitted subspheres
                        5. basisu (list): the loadings in the embedding feature space
                        6. mean (vector): PNS mean. This is the center of the distribution
                        7. itype (list): user-selected types (great sphere:2 or small sphere:1)
        """

        dist = []
        resmat = []
        orthaxis = []
        gsphere = []
        pvalues = []


        for i in range(dm-1):
            circle_type = 'SMALL' if self.itype == 1 else 'GREAT'
            print(str(dm-i) + '-sphere to ' + str(dm-i-1) + '-sphere by fitting a ' + circle_type +' sphere')
            center, r = self.get_subsphere(data, small_circle=(self.itype==1))
            curr_angle = np.matmul(center.T, data)
            res = np.arccos(np.clip(curr_angle, -1, 1)) - r
            orthaxis.append(center)

            dist.append(r)
            resmat.append(res.squeeze())

            nested_sphere = np.matmul(rotate_to_north_pole(center.squeeze()), data)
            data = nested_sphere[:dm-i, :] / np.sqrt(1-nested_sphere[dm-i, :] ** 2)[np.newaxis,:]
            gsphere.append(self.itype - 1)

        ## parameterize 1-sphere to angles
        if True: #nullspdim + 1 - (dm - 1) <= 0:
            s1_to_radian = np.arctan2(data[1, :], data[0, :])
            mean_theta, _ = self.geod_mean_s1(s1_to_radian.T)
            orthaxis.append(mean_theta)
            last_res = (s1_to_radian - mean_theta + np.pi) % (2*np.pi) - np.pi
            resmat.append(last_res)
        ## scale resmat according to the sizes of subspheres
        radii = [1.0]
        for i in range(1, dm):
            radii.append(np.prod(np.sin(dist[:i])))
        resmat = np.flipud(np.array(radii)[:, np.newaxis] * resmat)

        PNS = {'radii': radii, 'orthaxis': orthaxis, 'dist': dist, 'pvalues': pvalues, \
                'gsphere': gsphere, 'basisu': basisu, 'mean': [], 'itype': self.itype}
        PNS['mean'] = self.inv(np.zeros((dm, 1)), PNS)
        return (resmat, PNS)

    def geod_mean_sk(self, data, tol=1e-10):
        """
        Geodesic mean of data on S^k (Sphere) use Log map and Exp

        Args:
            data (numpy.ndarray): a matrix (k+1)-by-n: a column vector represents a point on S^k
            tol (float):  tolerance that stops the iteration

        Returns:
        vini (numpy.ndarray): A vector of dimension (k-1)-by-1, geodesic mean on the hypersphere S^(k-1)
        """
        vini = data[:, 0]
        diff = 1
        while dff > tol:
            rot = rotate_to_north_pole(vini)
            rot_data = rot @ data
            mean_in_tangent = np.mean(rot_data, axis=1)
            v_new = exp_north_pole(mean_in_tangent)
            pull_back_v_new = np.linalg.inv(rot) @ v_new
            diff = np.linalg.norm(pull_back_v_new - vini)
            vini = pull_back_v_new
        return vini
    def geod_mean_s1(self, theta):
        """
        Geodesic mean of data on S^1 (Circle) by S. Lu and V. Kulkarni
        method - gives all multiples of geodesic mean set.

        Args:
            theta (float): a column vector of angles
        
        Returns:
            geod_mean (numpy.ndarray): geodesic mean on S^1
            geod_var (numpy.ndarray):  geodesic variance on S^2
        """
        n = len(theta.squeeze())
        mean_cand = (abs(np.mean(theta)) + 2*np.pi*np.arange(n) / n) % (2*np.pi)
        theta = theta % (2*np.pi)
        geod_var = np.zeros((n, 1))
        for i in range(n):
            v = mean_cand[i]
            var1 = (theta - v) ** 2
            var2 = (theta - v + 2 * np.pi) ** 2
            var3 = (v - theta + 2 * np.pi) ** 2
            dist2 = np.min(np.vstack((var1[None,:], var2[None,:], var3[None,:])), axis=0)
            geod_var[i] = np.sum(dist2)
        ind = np.argmin(geod_var)
        geod_mean = mean_cand[ind] % (2*np.pi)
        geod_var = geod_var[ind] / n
        return geod_mean, geod_var
    def get_subsphere(self, data, small_circle=False):
        """
        The least square estimates of the best fitting subsphere
        to the data on the unit hyper-sphere.
        [center, r]= getSubSphere(data), with d x n data matrix with each
        column having unit length, returns the center and the
        radius.

        Args:
            data (numpy.ndarray): A 2D matrix of dimension d-by-N, where d is the number of features and N is the number of  cases
            small_circle (bool): True if the subsphere is parameterized by small circle

        Returns:
            center (numpy.ndarray): the vector of the center of the fitted subsphere
            r (float): the radius of the fitted subsphere
        """


        def obj_fun(center, r, data):
            """
            the objective function that we want to minimize: sum of squared distances
            from the data to the subsphere
            """
            test = np.matmul(center.T, data)
            test = np.clip(test, -1, 1)
            return np.mean((np.arccos(test) - r) ** 2)

        def est_subsphere(data, c0):
            tol = 1e-9
            cnt = 0
            err = 1
            d, n = data.shape
            g_now = 1e10
            center = None
            r = None
            while err > tol:
                c0 = c0 / np.linalg.norm(c0)
                rot = rotate_to_north_pole(c0)
                tp_data = log_north_pole(np.matmul(rot, data))
                new_center_tp, r = self.least_square_fit_sphere(tp_data, np.zeros(d-1), small_circle)
                if r > np.pi:
                    r = np.pi / 2
                    u, s, _ = scipy.linalg.svd(tp_data, lapack_driver='gesvd')
                    ## add minus sign to keep consistent with the results from MATLAB
                    last_singular_vect = u[:, -1]

                    new_center_tp = last_singular_vect * np.pi / 2
                new_center = exp_north_pole(x=new_center_tp[:, np.newaxis])
                center = np.matmul(np.linalg.inv(rot), new_center)
                g_next = obj_fun(center, r, data)
                err = abs(g_now - g_next)
                g_now = g_next
                c0 = center.squeeze()
                cnt += 1
                if cnt > 30:
                    print('Fit subspheres iteration reached 30th with residuals: {}'.format(err))
                    break
            return (g_now, center, r)

        if np.any(np.isnan(data)):
            #Y_hat, mu_hat, u, s, Vt = self.emsvd(data)
            data = np.nan_to_num(data)

        u, s, _ = scipy.linalg.svd(data, lapack_driver='gesvd')
        initial_center = u[:, -1]

        ### Zhiyuan: Keep initial_center in north hemisphere
        north_pole = np.zeros_like(initial_center)
        north_pole[-1] = 1
        # if np.inner(north_pole, initial_center) < 0:
        #     initial_center = -initial_center
        c0 = initial_center
        i1_save = est_subsphere(data, c0)

        pca = PCA()
        pca.fit(data.T)
        u = pca.components_.T
        ### Zhiyuan: Here find the last "effective" eigenvector of COVARIANCE matrix
        initial_center = u[:, -1]
        for i_vector in range(len(pca.explained_variance_) - 1, -1, -1):
            if pca.explained_variance_[i_vector] > 1e-15:
                initial_center = u[:, i_vector]
                break

        # if np.inner(north_pole, initial_center) < 0:
        #     initial_center = -initial_center

        c0 = initial_center
        i2_save = est_subsphere(data, c0)

        if i1_save[0] <= i2_save[0]:
            center = i1_save[1]
            r = i1_save[2]
        else:
            center = i2_save[1]
            r = i2_save[2]
        if r > np.pi / 2:
            center = -center
            r = np.pi - r
        return center, r
    # def geodesic_dist(self, r1, r2):
    #     """
    #     Geodesic distance

    #     Input r1, r2: n x 1 vector
    #     """
    #     k = (np.linalg.norm(r1)) ** 2 + (np.linalg.norm(r2)) ** 2
    #     theta = 2 * np.inner(r1, r2) / k
    #     if theta < -1:
    #         theta = -1
    #     elif theta > 1:
    #         theta = 1
    #     return np.abs(np.arccos(theta))

    def least_square_fit_sphere(self, data, initial_center=None, small_circle=False):
        """
        The least square estimates of the sphere to the data.
        the Levenberg-Marquardt method in Fletcher's modification
        (Fletcher, R., (1971): A Modified Marquardt Subroutine for
        Nonlinear Least Squares. Rpt. AERE-R 6799, Harwell)
        and implemented for MATLAB by M. Balda's "LMFnlsq.m"

        Args:
            data (numpy.ndarray): A 2D matrix of dimension d-by-N, where d is the number of features and N is the number of  cases
            initial_center (numpy.ndarray): The intial guess of the center
            small_circle (bool): True if the subsphere is parameterized by small circle

        Returns:
            center (numpy.ndarray): the vector of the center of the fitted subsphere
            r (float): the radius of the fitted subsphere

        """
        if initial_center is None:
            initial_center = np.mean(data, axis=1)

        def compute_residuals(x):
            x = x[:, np.newaxis]
            di = np.sqrt(np.sum((data - x) ** 2, axis=0))
            r = np.pi / 2

            if small_circle:
                r = np.sum(di) / len(di)
            di = di - r
            return di
        opt_solution = None
        opt_solution = opt.least_squares(compute_residuals, initial_center, method='lm', max_nfev=50, xtol=1e-15)

        # if small_circle:
        #     opt_solution = opt.least_squares(compute_residuals, initial_center, max_nfev=50, xtol=1e-9)
        # else:
        #     opt_solution = opt.least_squares(compute_residuals, initial_center, method='lm', max_nfev=50, xtol=1e-9)

        center = opt_solution.x
        di = np.sqrt(np.sum((data - center[:, np.newaxis]) ** 2, axis=0))
        if small_circle:
            r = np.mean(di)
        else:
            r = np.pi / 2
        return center, r

    @staticmethod
    def inv(resmat, coords):
        """
        Invert PNS that converts Euclidean representation from PNS to coords in extrinsic coords

        Args:
            resmat (numpy.ndarray): Euclideanized features of dimension (k-1)-by-k from PNS.fit
            coords (tuple): PNS configurations (subspheres) from PNS.fit

        Returns:
            T (numpy.ndarray): A d-by-N matrix representing with extrinsic coords, where d is the number of features in the embedding space and N is the number of cases
        """
        d, n = resmat.shape
        ns_orthaxis = np.flipud(np.array(coords['orthaxis'][:-1], dtype="object"))
        ns_radius = np.flipud(np.array(coords['dist'], dtype="object"))
        geodmean = coords['orthaxis'][-1]

        res = resmat / np.flipud(coords['radii'])[:, np.newaxis]

        ## convert coords for S^1 (i.e., a circle)
        ## by adding the mean value to each residual (also degrees)
        if d > 0:
            T = np.vstack((np.cos(geodmean + res[0, :]), np.sin(geodmean + res[0, :])))

        ## S^1 coords to S^2
        if d > 1:
            prev_T = np.vstack((np.cos(geodmean + res[0, :]), np.sin(geodmean + res[0, :])))
            factor = np.sin(ns_radius[0] + res[1, :])
            sin_theta = factor[np.newaxis, :] * prev_T
            cos_theta = np.cos(ns_radius[0] + res[1, :])
            curr_T = np.vstack((sin_theta, cos_theta))
            rot_mat = rotate_to_north_pole(ns_orthaxis[0].squeeze())
            T = np.matmul(rot_mat.T, curr_T)

        ## S^2 to S^d
        if d > 2:
            for i in range(d-2):
                rot_mat = rotate_to_north_pole(ns_orthaxis[i+1].squeeze())
                factor = np.sin(ns_radius[i+1] + res[i + 2, :])
                sin_theta = factor[np.newaxis, :] * T
                cos_theta = np.cos(ns_radius[i+1] + res[i+2, :])
                curr_T = np.vstack((sin_theta, cos_theta))
                T = np.matmul(rot_mat.T, curr_T)
        np_basisu = np.array(coords['basisu'])
        if np_basisu.size != 0:
            ### Relate to the original feature space
            T = np.matmul(np_basisu, T)
        return T