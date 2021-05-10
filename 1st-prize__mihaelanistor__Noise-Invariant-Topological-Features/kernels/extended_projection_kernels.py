import numpy as np

'''
The proxy functions implement the actual kernels, while their 
non-proxy coutnerparts to return the gram_matrix,which is needed 
in SVC's kernel or fit. This is a technical quirk and you need to 
use a proxy kernel or construct the gram matrix by hand.

Explanation: 
https://stackoverflow.com/questions/43994505/i-am-trying-to-implement-svm-in-scikit-learn-with-custom-rbf-kernel-but-it-is-s

The following 3 kernels are implemented from: 
https://arxiv.org/abs/1807.10400
'''

def GrassmannianRBFKernel_proxy(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    beta = 1

    output = np.exp(-beta * np.linalg.norm(np.matmul(Y1.T, Y2), 'fro') ** 2)

    return output

def GrassmannianRBFKernel(Y1: np.ndarray, Y2: np.ndarray, K=GrassmannianRBFKernel_proxy) -> np.ndarray:
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

def GrassmannianProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    output = np.linalg.norm(np.matmul(Y1.T, Y2), 'fro') ** 2

    return output

def GrassmannianProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=GrassmannianProjectionKernel_proxy) -> np.ndarray:
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

'''
The following 6 kernels are implemented from:
https://papers.nips.cc/paper/2008/hash/e7f8a7fb0b77bcb3b283af5be021448f-Abstract.html
'''

def orthonormalization(Y1: np.ndarray) -> np.ndarray:
    output = np.matmul(Y1, 1 / np.sqrt(np.matmul(Y1.T, Y1)))

    return output

def LinearProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray):
    output = np.trace(np.matmul(np.matmul(orthonormalization(Y1).T, orthonormalization(Y1)), 
                                np.matmul(orthonormalization(Y2).T, orthonormalization(Y2))))

    return output

def LinearProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=LinearProjectionKernel_proxy):
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

def LinearScaledProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray):
    output = np.trace(np.matmul(np.matmul(orthonormalization(Y1).T, orthonormalization(Y2)), np.matmul(Y2.T, Y1)))

    return output

def LinearScaledProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=LinearScaledProjectionKernel_proxy):
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

def LinearSpherisedProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    output = LinearScaledProjectionKernel(Y1, Y2) * \
             (1 / np.sqrt(LinearScaledProjectionKernel(Y1, Y1))) * \
             (1 / np.sqrt(LinearScaledProjectionKernel(Y2, Y2)))

    return output

def LinearSpherizedProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=LinearSpherisedProjectionKernel_proxy):
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

def AffineProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    u1 = Y1.mean(0)
    component1 = np.matmul(orthonormalization(Y1), orthonormalization(Y1).T)
    u2 = Y2.mean(0)
    component2 = np.matmul(orthonormalization(Y2), orthonormalization(Y2).T)

    output = LinearProjectionKernel(Y1, Y2) + \
             u1.T * (np.identity(component1.shape[0]) - component1) * (np.identity(component2.shape[0]) - component2) * u2

    return output

def AffineProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=AffineProjectionKernel_proxy) -> np.ndarray:
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

def AffineScaledProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    u1 = Y1.mean(0)
    component1 = np.matmul(orthonormalization(Y1), orthonormalization(Y1).T)
    u2 = Y2.mean(0)
    component2 = np.matmul(orthonormalization(Y2), orthonormalization(Y2).T)
    
    output = LinearScaledProjectionKernel(Y1, Y2) + \
             u1.T * (np.identity(component1.shape[0]) - component1) * (np.identity(component2.shape[0]) - component2) * u2

    return output

def AffineScaledProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=AffineScaledProjectionKernel_proxy) -> np.ndarray:
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix

def AffineSpherisedProjectionKernel_proxy(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    output = AffineScaledProjectionKernel(Y1, Y2) * \
             (1 / np.sqrt(AffineScaledProjectionKernel(Y1, Y1))) * \
             (1 / np.sqrt(AffineScaledProjectionKernel(Y2, Y2)))

    return output

def AffineSpherisedProjectionKernel(Y1: np.ndarray, Y2: np.ndarray, K=AffineSpherisedProjectionKernel_proxy) -> np.ndarray:
    gram_matrix = np.zeros((Y1.shape[0], Y2.shape[0]))

    for i, x in enumerate(Y1):
        for j, y in enumerate(Y2):
            gram_matrix[i, j] = K(x, y)

    return gram_matrix
