import numpy as np
from pyPNS import PNS

import scipy.io
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

### Read toy example data which distributed along a small circle  on S^2
small_circle_data = scipy.io.loadmat('../data/toy_example_small_circle.mat')
data = small_circle_data['data']

### Fit PNS to data
pns_model = PNS(data, itype=9)
pns_model.fit()
resmat, PNS_coords = pns_model.output

### Invert the resmat
invert_pns = PNS.inv(resmat, PNS_coords)
print(data)
print(invert_pns)

### Plot the graph for PNS vs. PCA
pns_pca = PCA()
pns_pca.fit(resmat.T)
explained_variance_ratio = pns_pca.explained_variance_ratio_
num_components_pns = len(explained_variance_ratio)
pns_cum_ratio = []
for i in range(1, num_components_pns + 1):
    pns_cum_ratio.append(np.sum(explained_variance_ratio[:i]))

pure_pca = PCA()
pure_pca.fit(data.T)
pca_explained_ratio = pure_pca.explained_variance_ratio_
num_components_pca = len(pca_explained_ratio)
pca_cum_ratio = []
for i in range(1, num_components_pca + 1):
    pca_cum_ratio.append(np.sum(pca_explained_ratio[:i]))

fig, axs = plt.subplots(1, 2)

axs[0].bar(range(1, num_components_pns + 1), explained_variance_ratio)
axs[0].plot(np.arange(1, num_components_pns + 1), pns_cum_ratio, 'r-x')
axs[0].set_title('PNS results')
axs[0].set_ylabel('Explained variance ratio')
axs[0].set_xlabel('# Principal components')


axs[1].bar(range(1, num_components_pca + 1), pca_explained_ratio)
axs[1].plot(range(1, num_components_pca + 1), pca_cum_ratio, 'r-x')
axs[1].set_title('PCA results')
axs[1].set_xlabel('# Principal components')

plt.show()

print('Done')