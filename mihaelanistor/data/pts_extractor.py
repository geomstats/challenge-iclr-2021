import numpy as np
from gtda.utils.validation import check_diagrams
from scipy import stats
from scipy.sparse import linalg
import warnings

class ExtractPTS:
    """ Extracts Topological Persistence Signatures from Persistence Diagrams as
    in the following paper:
    Som, A., Thopalli, K., Ramamurthy, K.N., Venkataraman, V., Shukla, A. and 
    Turaga, P., Perturbation Robust Representations of Topological Persistence 
    Diagrams.
    https://arxiv.org/pdf/1807.10400.pdf
    Adapted from the official repository:
    https://github.com/anirudhsom/Perturbed-Topological-Signature

    A set of Perturbed PDs is created by applying Gaussian noise on the initial 
    PD with transformed axis. The Perturbed PDs are then converted to 2D PDFs by
    fitting a Gaussian kernel function at each point in the PD, normalizing the
    2D surface and discretizing the surface over a `x1` x `x2` grid. The 2D PDFs
    are then mapped to the Grassmanian manifold. The largest singular vectors
    are extracted using SVD.

    Parameters
    ----------
    n_perturbations : int, optional, default: ``40``
        The number of perturbations extracted from the initial PD.
    
    max_displacement : float, optional, default: ``0.05``
        The displacement done to the points from the initial PD.

    x1 : int, optional, default: ``50``
        Number of lines for the discretization of the PDF.

    x2 : int, optional, default: ``50``
        Number of columns for the discretization of the PDF.

    sigma : float, optional, default: ``0.04``
        The standard deviation (bandwidth parameter) for the Gaussian used to
        convert the Perturbed PDs to PDFs.

    subspace_dimension : int, optional, default: ``10``
        Subspace dimension for the Grassmann manifold.

    tries : int, optional, default: ``50``
        How many times to try again if the transformation fails.
    """

    def __init__(
        self,
        n_perturbations=40,
        max_displacement=0.05,
        x1=50,
        x2=50,
        sigma=0.04,
        subspace_dimension=10,
        tries=50
    ):
        self.n_perturbations = n_perturbations
        self.max_displacement = max_displacement
        self.x1 = x1
        self.x2 = x2
        self.sigma = sigma
        self.subspace_dimension = subspace_dimension
        self.tries = tries

    def fit(self, X, y=None):
        """ Does nothing, returns an unchanged estimator.
        
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_diagrams(X)
        self.__is_fitted = True
        return self

    def transform(self, X, y=None):
        """ Transforms a PDs into PTS representations.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            We assume that the PDs are normalized. The normalization scheme
            in the official implementation is PD_norm = PD/max(PD).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        pts_repr : ndarray of shape 
            (n_samples, n_features, `subspace_dimension`)
            PTS features.

        """
        
        Xs = check_diagrams(X, copy=True)
        pts_list = []
        n_pds = Xs.shape[0]

        for pd in Xs:

            perturbed_pds = self.make_perturbation(pd)
            pdfs = self.make_pdfs(perturbed_pds)
            while self.subspace_dimension >= len(pdfs) and self.tries > 0:
                perturbed_pds = self.make_perturbation(pd)
                pdfs = self.make_pdfs(perturbed_pds)
                self.tries -= 1
            else:
                if self.subspace_dimension >= len(pdfs):
                    raise ValueError("Cannot extract PTS. Please try again")

            manifold_point = self.map_to_manifold(pdfs)
            manifold_point = np.expand_dims(manifold_point, axis=0)
            pts_list.append(manifold_point)

        pts_repr = np.concatenate(pts_list, axis=0)
        return pts_repr

    def make_perturbation(self, X):
        """ Perturbs a single PD and returns a list size `n_perturbations` + 1 
        PDs.

        Parameters
        ----------
        X : ndarray of shape (n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            We assume that the PDs are normalized. The normalization scheme
            in the official implementation is PD_norm = PD/max(PD).
        
        Returns
        -------
        perturbed_pds : list with `n_perturbations` + 1 PDs of shape 
            (n_features, 3)

        """

        Xs = X.copy()
        # We expect a single PD
        if len(Xs.shape) > 2:
            Xs = np.squeeze(Xs, axis=0)

        n_points = X.shape[0]
        perturbed_pds = []

        # Transform the axes of the PD: (b, d) -> ( (b+d)/2, d-b )
        b = 0.5 * (Xs[:, :1] + Xs[:, 1:2])
        d = Xs[:, 1:2] - Xs[:, :1]
        h = np.squeeze(Xs[:, 2:3], axis=1)

        Xs[:, :1] = b
        Xs[:, 1:2] = d

        # Append the unperturbed PD to the list
        perturbed_pds.append(Xs)

        # Create randomly perturbed PDs
        random_pert_b = np.random.random((n_points, self.n_perturbations))
        random_pert_d = np.random.random((n_points, self.n_perturbations))

        random_pert_b = random_pert_b * 2 * self.max_displacement
        random_pert_d = random_pert_d * 2 * self.max_displacement

        random_pert_b = random_pert_b - self.max_displacement
        random_pert_d = random_pert_d - self.max_displacement

        b_perturbed = b + random_pert_b
        d_perturbed = d + random_pert_d

        for i in range(self.n_perturbations):
            b_i = b_perturbed[:, i]
            d_i = d_perturbed[:, i]
            h_i = h.copy()
            # Remove entries with negative birth
            b_i = b_i[np.where(b_i >= 0)]
            d_i = d_i[np.where(b_i >= 0)]
            h_i = h_i[np.where(b_i >= 0)]
            # Remove entries with negative death
            b_i = b_i[np.where(d_i >= 0)]
            d_i = d_i[np.where(d_i >= 0)]
            h_i = h_i[np.where(d_i >= 0)]
            # Remove entries with birth >= 1
            b_i = b_i[np.where(b_i <= 1)]
            d_i = d_i[np.where(b_i <= 1)]
            h_i = h_i[np.where(b_i <= 1)]
            # Remove entries with death >= 1
            b_i = b_i[np.where(d_i <= 1)]
            d_i = d_i[np.where(d_i <= 1)]
            h_i = h_i[np.where(d_i <= 1)]

            perturbed_pd = np.column_stack((b_i, d_i, h_i))
            perturbed_pds.append(perturbed_pd)

        return perturbed_pds

    def make_pdfs(self, perturbed_pds):
        """ Converts the Perturbed PDs to 2D surfaces using a Gaussian kernel
        function and discretizes into a `x1` x `x2` grid.

        Parameters
        ----------
        perturbed_pds : list of `n_perturbations` + 1 PDs of shape 
            (n_features, 3).
        
        Returns
        -------
        pdfs : list of `n_perturbations` + 1 discretized PDFs of shape
            ((`x1`+1) * (`x2`+1),).

        """

        pdfs = []
        x1 = np.arange(0, 1.01, 1 / self.x1)
        x2 = np.arange(0, 1.01, 1 / self.x2)
        X1, X2 = np.meshgrid(x1, x2)
        positions = np.vstack([X1.ravel(), X2.ravel()])

        heatmap = np.zeros((self.x1 + 1, self.x2 + 1))

        # print('-'*20)
        for idx, pd in enumerate(perturbed_pds):
            # print('idx:', idx)
            pd_t = pd[:, :2].T.copy()
            try:
                # If the PD has few points, the following problem occurs:
                # try-catch wrapper for the following cases:
                # 1. pd_t is singular
                # 2. positions matrix is not positive definite 
                # 3. positions is null
                # The pipeline should not be affected due to the SVD.
                # Issue: It will crash if the number of vectors extracted is 
                # less than `subspace_dimension`
                kernel = stats.gaussian_kde(pd_t, self.sigma)
                transformed = kernel(positions)
                Z = np.reshape(transformed.T, X1.shape)
                heatmap = heatmap + Z
                pdf = heatmap.reshape((X1.shape[0] * X2.shape[1]))
            except Exception as e:
                # warnings.warn(f"{e}") # we check at the end if we could get a minimum of subspace_dimension pdfs
                continue

            pdf = pdf / np.sum(pdf)
            pdfs.append(pdf)

        return pdfs

    def map_to_manifold(self, pdfs):
        """ Converts the Perturbed PDs to 2D surfaces using a Gaussian kernel
        function and discretizes into a `x1` x `x2` grid.

        Parameters
        ----------
        pdfs : list of `n_perturbations` + 1 discretized PDFs of shape
            ((`x1`+1) * (`x2`+1),).

        Returns
        -------
        U : PTS representation of shape 
            ((`x1`+1) * (`x2`+1), subspace_dimension).
            Contains the largest `subspace_dimension` orthonormal singular 
            vectors.

        """
        column_pdfs = np.column_stack(pdfs)
        U, _, _ = linalg.svds(column_pdfs, k=self.subspace_dimension)
        return U