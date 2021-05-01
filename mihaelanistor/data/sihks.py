import numpy as np

from scipy.io import loadmat
from scipy.fft import fft


def SIHKS(evecs, evals, t0=0.01, time_scale=15, alpha1=2, n_functions=17):
    n_vertices = evecs.shape[0]

    tau = np.linspace(start=0, stop=time_scale, num=int(time_scale/(1/16)+1))
    t = t0 * alpha1 ** tau

    hks = np.zeros((n_vertices, len(tau)))  # (1002, 241)

    for i in range(len(tau)):
        sum1 = np.multiply(-np.log(alpha1) * evecs ** 2,
                           np.tile(np.multiply(t[i] * evals.T, np.exp(-(t[i] * evals.T))), reps=(n_vertices, 1))).sum(axis=1)
        
        sum2 = np.multiply(evecs ** 2, np.tile(np.exp(-t[i] * evals.T), reps=(n_vertices, 1))).sum(axis=1)
        
        hks[:, i] = np.divide(sum1, sum2)
    
    shks = np.zeros((n_vertices, len(tau) - 1))
    for i in range(len(tau) - 1):
        shks[:, i] = hks[:, i + 1] - hks[:, i]

    sihks = np.zeros((n_vertices, len(tau) - 1))
    for i in range(n_vertices):
        sihks[i, :] = np.abs(fft(shks[i, :]))

    return sihks[:, :n_functions]


def make_point_clouds(vertices, temperature):
    """[summary]

    Parameters
    ----------
    vertices : ndarray of shape (n_vertices, 3)
        Vertices of the mesh as 3D points.
    temperature : ndarray of shape (n_vertices, n_functions)
        A collection of functions defined on the vertices of the mesh, such as SIHKS or other spectral descriptor.

    Returns
    -------
    point_clouds : ndarray of shape (n_functions, n_vertices, 4)
        Collection of point clouds formed by concatenating the vertex coordinates and the corresponding
        temperature for each given function.
    """
    n_vertices = vertices.shape[0]
    n_functions = temperature.shape[1]

    # Repeat points n_function times [n_functions, n_vertices, 3]
    vertices = np.tile(vertices, reps=(n_functions, 1))
    vertices = vertices.reshape(n_functions, n_vertices, 3)

    # Reshape temperature [n_functions, n_vertices, 1]
    temperature = np.expand_dims(temperature.T, axis=-1)

    # Concatenate coordinates and temperature
    point_clouds = np.concatenate([vertices, temperature], axis=-1)

    return point_clouds

