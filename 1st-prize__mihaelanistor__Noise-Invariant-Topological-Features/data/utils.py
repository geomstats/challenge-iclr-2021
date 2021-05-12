import trimesh
import sys
import tempfile
import pickle
import numpy as np

from tqdm import tqdm
from os import listdir, mkdir
from os.path import isfile, exists, join, splitext, dirname
from PIL import Image

sys.path.append(join(dirname(__file__), 'ShapeDNA'))
from laplace_beltrami import laplace_beltrami_eigenvalues
from data.sihks import SIHKS


def load_meshes(dataset_path):
    """ Loads all meshes from given directory.

    Parameters
    ----------
    dataset_path : str
        Path to the directory containing the meshes.

    Returns
    -------
    meshes: 
        Dictionary of meshes (as trimesh.base.Trimesh) indexed by name.
    """
    meshes = {}

    # Load all meshes
    for file in tqdm(listdir(dataset_path), desc='Loading meshes'):
        mesh_path = join(dataset_path, file)
        if isfile(mesh_path):
            mesh = trimesh.load(mesh_path)
            meshes[file.split('.')[0]] = mesh

    return meshes

def simplify_meshes(dataset_path, output_path=None, out_type='obj', num_faces=2000):
    """ Simplifies each 3D mesh from the original SHREC dataset to a given number of faces.
    Saves a copy of the simplified meshes on disk in `output_path` is provided. 

    Parameters
    ----------
    dataset_path : str
        Path to the original dataset containing the meshes.
    output_path : str
        Path to the output directory where the simplified meshes are exported.
    out_type : str, optional
        Format of the exported mesh, by default 'obj'.
    num_faces : int, optional
        Number of faces of the simplified mesh, by default 2000.

    Returns
    -------
    meshes
        Dictionary of simplified meshes (as trimesh.base.Trimesh) indexed by name.
    """
    if not exists(output_path):
        mkdir(output_path)

    meshes = {}

    # Process all meshes in original dataset
    for file in tqdm(listdir(dataset_path), desc='Simplifying meshes to {} faces'.format(num_faces)):
        mesh_path = join(dataset_path, file)
        if isfile(mesh_path):
            # Load mesh
            mesh = trimesh.load(mesh_path)
            
            # Simplify mesh
            mesh = mesh.simplify_quadratic_decimation(face_count=num_faces)
            meshes[file.split('.')[0]] = mesh
            
            # Export mesh
            filename, _ = splitext(file)
            mesh.export(join(output_path, filename + '.' + out_type))

    return meshes


def compute_eigen(dataset_path, output_path, num_eigen=150):
    """ Computes the eigenvalues and eigenfunctions of the Laplace-Beltrami
    operator for each 3D mesh in the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset containing the meshes.
    output_path : str
        Path to the output directory where the spectrum is saved.
    num_eigen : int, optional
        The number of eigenvalues and eigenvectors to compute, by default 150.
    
    Returns
    -------
    eigen
        Dictionary of eigenvalues and eigenvectors indexed by mesh name.
    """
    if not exists(output_path):
        mkdir(output_path)
    
    eigen = {}

    for file in tqdm(listdir(dataset_path)):
        pickle_path = join(output_path, splitext(file)[0] + '_eigen.pckl') 

        # Prevent recomputing
        if exists(pickle_path):
            # print('Mesh {} already computed!'.format(file))
            with open(pickle_path, 'rb') as f:
                pickle_dict = pickle.load(f)
                eigen[file.split('.')[0]] = pickle_dict
            continue

        mesh_path = join(dataset_path, file)
        if isfile(mesh_path):
            try:
                _, evals, evecs, points, A, M = laplace_beltrami_eigenvalues(mesh_path, n_evals=(0, num_eigen))
            except Exception:
                print('File {} skipped.'.format(file))
                continue

            # Save eigenvalues and eigenvector as pickle
            pickle_dict = {'evals': evals.reshape(-1, 1), 'evecs': evecs}
            filename, _ = splitext(file)
            with open(join(output_path, filename + '_eigen.pckl'), 'wb') as f:
                pickle.dump(pickle_dict, f)
            
            eigen[file.split('.')[0]] = pickle_dict
    
    return eigen


def compute_sihks(eigen, output_path):
    if not exists(output_path):
        mkdir(output_path)

    sihks = {}

    for mesh_id in tqdm(eigen.keys(), desc='Computing SIHKS'):
        pickle_path = join(output_path, mesh_id + '_sihks.pckl')

        # Prevent recomputing
        if exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                pkle = pickle.load(f)
                sihks[mesh_id] = pkle
        else:
            sihks[mesh_id] = SIHKS(eigen[mesh_id]['evecs'], eigen[mesh_id]['evals'])
            with open(pickle_path, 'wb') as f:
                pickle.dump(sihks[mesh_id], f)

    return sihks


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


def load_class_labels(labels_path='/home/armand/repos/ICLR_Challenge/NonRigid/For_evaluation/test.cla'):
    class_labels = {}
    labels = {}
    class_names = []
    
    with open(labels_path, 'r') as f:
        # Skip header
        next(f)
        next(f)

        crt_class = None
        for line in f:
            line = line.strip()

            # Skip empty lines
            if line == '':
                continue

            line = line.split(' ')
            if len(line) == 3:
                crt_class = line[0]
                class_labels[crt_class] = []
            else:
                class_labels[crt_class].append('T{}'.format(line[0]))

    for it, class_name in enumerate(class_labels.keys()):
        class_names.append(class_name)
        for mesh_id in class_labels[class_name]:
            labels[mesh_id] = it
    
    return labels, class_names


def render_mesh(mesh):
    with tempfile.NamedTemporaryFile(suffix='.png') as file_obj:
        scene = mesh.scene()
        scene.save_image(resolution=(1080,1080))
        file_obj.seek(0)
        data = file_obj.read()
        rendered = Image.open(trimesh.util.wrap_as_stream(data))
        rendered.show()

