
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
import math
import random
import warnings
import persim
from scipy.spatial import distance
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance


def fpd_cluster(data, c, hom_dimension, metric='wasserstein', verbose=False, max_iter=10, frand='no', fuzzy=True):
    # Compute topological fuzzy clusters of a collection of point clouds
    #
    # INPUTS
    # data - collection of datasets
    # c - number of clusters
    # verbose - True or False to give iteration information
    # max_iter - max number of iterations to compute
    # p - dimension of persistence diagram (0=connected components, 1=holes, 2=voids, etc.) 
    # max_range - Max distance to consider between points for VR complex 
    # T - replace points at infinity with large hyperparameter T
    # frand - optional Fuzzy RAND reference matrix
    # fuzzy - fuzzy clustering if True, hard clustering if False
    # (if unsure of value for max_range or T, set as the furthest distance between two points)
    #
    # OUTPUTS
    # r - membership values
    # M - list of cluster centres
    # frand_indices - returns Fuzzy RAND index at each iteration (if reference matrix given)

    VR = VietorisRipsPersistence(homology_dimensions=[hom_dimension])
    diagrams = VR.fit_transform(data)
    # diagrams = np.delete(diagrams, axis=2, obj=2)
    r, M = pd_fuzzy(diagrams, c, verbose, max_iter, frand=frand, fuzzy=fuzzy, metric=metric)

    return r, M


def pd_fuzzy(D, c, verbose=False, max_iter=10, frand='no', fuzzy=True, metric='wasserstein'):
    # computes fuzzy clusters of persistence diagrams
    #
    # INPUTS
    # D - list of persistence diagrams
    # c - number of clusters
    # verbose - True or False to give iteration information
    # max_iter - max number of iterations to compute
    # frand - optional Fuzzy RAND reference matrix
    # fuzzy - fuzzy clustering if True, hard clustering if False
    #
    # OUTPUTS
    # r - membership values
    # M - list of cluster centres
    # frand_indices - returns Fuzzy RAND index at each iteration (if reference matrix given)

    if max_iter == 0:
        exit('Maximum iterations is zero')

    # D = add_diagonals(D)
    M = init_clusters(D, c)
    M = np.array(M)

    n = len(D)
    m = len(D[0])

    # J_new = 2 * epsilon
    # J_prev = 0

    frand_indices = []
    counter = 0
    while counter < max_iter:
        if verbose:
            print("Fuzzy iteration: " + str(counter))
        counter += 1

        # update membership values
        # distance = PairwiseDistance(metric=metric)
        # distance.fit(M)
        # W = distance.transform(D)

        W = calc_W(D, M)
        r = calc_r(W, fuzzy)

        if verbose:
            J_temp = J(r, D, M, metric)
            print(" -- Update r -- ")
            print("   J(r, M) = " + str(J_temp))
        if frand != 'no':
            frand_t = fuzzy_rand(r.T, frand)
            print("   Fuzzy RAND = " + str(frand_t))
            frand_indices.append(frand_t)
        if verbose:
            print(" -- Update M -- ")

        # update cluster centres
        for k in range(c):
            M[k], _ = calc_frechet_mean(D, r, k, verbose)

        # compute J
        # J_prev = J_new
        # J_new = J(r, D, M)
        if verbose:
            J_new = J(r, D, M, metric)
            print("   J(r, M) = " + str(J_new))
        if frand != 'no':
            frand_t = fuzzy_rand(r.T, frand)
            print("   Fuzzy RAND = " + str(frand_t))
        if verbose and (frand == 'no'):
            print()

    if frand == 'no':
        return r, M
    else:
        return r, M, frand_indices


def calc_r(W, fuzzy=True, check_zero=False):
    # calculate membership values
    # inputs: array of Wasserstein distances W
    # returns: array r of membership values r[j][k]
    n = np.shape(W)[0]
    c = np.shape(W)[1]
    r = np.zeros((n, c))

    if check_zero:
        for j in range(n):
            for k in range(c):
                if W[j][k] == 0:
                    W[j][k] = 0.00001

    for j in range(n):
        for k in range(c):
            sum = 0
            for l in range(c):
                sum += W[j][k] / W[j][l]
            r[j][k] = 1 / sum

    if not fuzzy:
        for j in range(n):
            max_ind = np.argmax(r[j])
            for k in range(c):
                if k == max_ind:
                    r[j][k] = 1
                else:
                    r[j][k] = 0

    return r


def calc_W(D, M):
    # calculate pairwise Wasserstein distances
    # inputs: array of diagrams D, array of centres M
    # returns: array W with w[j][k] = W_2(D_j, M_k)
    n = len(D)
    c = len(M)

    # W[j,k] = W_2(D_j, M_k)
    W = np.zeros((n, c))

    for j in range(n):
        for k in range(c):
            wass = calc_wasserstein(D[j], M[k])
            if wass != 0:
                W[j][k] = wass
            else:
                W[j][k] = 0.001
    return W


def calc_wasserstein(Dj, Mk):
    # calculates the 2-Wasserstein L2 distance between two diagrams
    # inputs: diagram Dj, centre Mk
    # returns: W_2(Dj, Mk)
    m = len(Dj)
    c = calc_cost_matrix(Dj, Mk)
    X = hungarian(c)
    total = 0
    for i in range(m):
        total += c[X[0][i]][X[1][i]]
    return math.sqrt(total)


def calc_cost_matrix(Dj, Mk):
    # calculates the cost matrix for optimal transport problem
    # inputs: diagram Dj, cluster centre Mk
    # returns: cost matrix c
    m = len(Dj)
    if m != len(Mk):
        exit("Incompatible diagram size in calc_cost_matrix: " + str(len(Dj)) + " and " + str(len(Mk)))

    c = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            # both off-diagonal
            if Dj[i][1] != 0 and Mk[j][1] != 0:
                c[i][j] = (Dj[i][0]-Mk[j][0])**2 + (Dj[i][1]-Mk[j][1])**2
            # only Dj[i] off-diagonal
            elif Dj[i][1] != 0 and Mk[j][1] == 0:
                c[i][j] = ((Dj[i][1] - Dj[i][0]) * 1/math.sqrt(2))**2
            # only Mk[j] off-diagonal
            elif Dj[i][1] == 0 and Mk[j][1] != 0:
                c[i][j] = ((Mk[j][1] - Mk[j][0]) * 1/math.sqrt(2))**2

    return c


def calc_wasserstein(Dj, Mk):
    # calculates the 2-Wasserstein L2 distance between two diagrams
    # inputs: diagram Dj, centre Mk
    # returns: W_2(Dj, Mk)
    m = len(Dj)
    c = calc_cost_matrix(Dj, Mk)
    X = hungarian(c)
    total = 0
    for i in range(m):
        total += c[X[0][i]][X[1][i]]
    return math.sqrt(total)


def calc_frechet_mean(D, r, k, verbose):
    # computes the weighted frechet mean of D with weights r[.][k]
    # inputs: diagrams D, membership values r, centre index k, verbose
    # returns: weighted frechet mean y, optimal pairings x
    ####
    DGM_DIM = 3  # length of point in pd
    ####

    n = len(D)
    m = len(D[0])
    # initialise to random diagram in D
    random.seed(0)
    M_update = D[random.randint(0, n-1)]

    # first run to find matching
    matching = []
    for j in range(n):
        c = calc_cost_matrix(M_update, D[j])
        x_indices = hungarian(c)
        matching.append(x_indices)

    # loop until stopping condition is found
    counter2 = 0

    while True:
        counter2 += 1

        # update matched points
        x = np.zeros((n, m, DGM_DIM))
        for j in range(n):
            for i in range(m):
                index = matching[j][1][i]
                x[j][i] = D[j][index]

        # generate y to return
        y = np.zeros((m, DGM_DIM))

        # loop over each point
        for i in range(m):
            # calculate w and w_\Delta
            r2_od = 0
            r2x_od = [0, 0]
            for j in range(n):
                if x[j][i][1] != 0:
                    r2_od += r[j][k]**2
                    r2x_od[0] += r[j][k]**2 * x[j][i][0]
                    r2x_od[1] += r[j][k]**2 * x[j][i][1]

            # if all points are diagonals
            if r2_od == 0:
                # then y[i] is a diagonal
                y[i] = [0, 0]

            # else proceed
            else:
                w = [r2x_od[0]/r2_od, r2x_od[1]/r2_od]
                w_delta = [(w[0]+w[1])/2, (w[0]+w[1])/2]

                r2_d = 0
                r2_w_delta = [0, 0]
                for j in range(n):
                    if x[j][i][1] == 0:
                        r2_d += r[j][k] ** 2
                        r2_w_delta[0] += r[j][k]**2 * w_delta[0]
                        r2_w_delta[1] += r[j][k]**2 * w_delta[1]

                # calculate weighted mean
                y[i][0] = (r2x_od[0] + r2_w_delta[0]) / (r2_od + r2_d)
                y[i][1] = (r2x_od[1] + r2_w_delta[1]) / (r2_od + r2_d)

        old_matching = matching.copy()
        matching = []
        for j in range(n):
            c = calc_cost_matrix(y, D[j])
            x_indices = hungarian(c)
            matching.append(x_indices)

        comparison = (np.array(matching) == np.array(old_matching))
        if comparison.all():
            if verbose:
                print("      Frechet iterations for M_" + str(k) + ": " + str(counter2))
            return y, x


def init_clusters(D, c):
    # initialise cluster centres to Frechet mean of two diagrams
    # inputs: diagrams D, number of clusters c
    # outputs: initialised cluster centres M
    M = []
    ones = np.ones((len(D)+1, c+1))
    for i in range(c):
        diagram, _ = calc_frechet_mean([D[i], D[i+1]], ones, i, verbose=False)
        M.append(diagram)

    return M


def J(r, D, M, metric):
    # computes the cost function J
    # inputs: membership values r, diagrams D, cluster centres M
    # returns: clustering cost

    # distance = PairwiseDistance(metric=metric)
    # distance.fit(M)
    # W = distance.transform(D)

    W = calc_W(D, M)

    n = np.shape(W)[0]
    c = np.shape(W)[1]

    sum = 0
    for j in range(n):
        for k in range(c):
            sum += r[j][k]**2 * W[j][k]**2

    return sum


def fuzzy_rand(Q, R):
    # Computes the Fuzzy Rand matrix (Campello, 2007) 
    # Inputs: Membership matrix Q, reference matrix R
    # Output: Fuzzy Rand index

    # R is v x N, v = number of cluster, N = number of data objects
    # R[i][j] is how much cluster j is associated with diagram i

    v = len(Q)      # number of clusters
    k = v           # number of true classes
    N = len(Q[0])   # number of persistence diagrams

    V = np.zeros((N, N))
    X = np.zeros((N, N))
    Y = np.zeros((N, N))
    Z = np.zeros((N, N))

    for j1 in range(N):
        for j2 in range(N):
            # Compute V
            min_values = []
            for i in range(k):
                min_values.append(min(R[i][j1], R[i][j2]))

            V[j1][j2] = max(min_values)

            # Compute X
            min_values = []
            for i1 in range(k):
                for i2 in range(k):
                    if i1 != i2:
                        min_values.append(min(R[i1][j1], R[i2][j2]))

            X[j1][j2] = max(min_values)

    for j1 in range(N):
        for j2 in range(N):
            # compute Y
            min_values = []
            for l in range(v):
                min_values.append(min(Q[l][j1], Q[l][j2]))

            Y[j1][j2] = max(min_values)

            # Compute Z
            min_values = []
            for l1 in range(v):
                for l2 in range(v):
                    if l1 != l2:
                        min_values.append(min(Q[l1][j1], Q[l2][j2]))

            Z[j1][j2] = max(min_values)

    a, b, c, d = 0, 0, 0, 0

    for j2 in range(1, N):
        for j1 in range(j2):
            a += min(V[j1][j2], Y[j1][j2])
            b += min(V[j1][j2], Z[j1][j2])
            c += min(X[j1][j2], Y[j1][j2])
            d += min(X[j1][j2], Y[j1][j2])

    return (a+d)/(a+b+c+d)
