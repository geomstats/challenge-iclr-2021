import numpy as np
from matplotlib import image
import os
from scipy import ndimage
import matplotlib.pyplot as plt

im = image.imread("../data/mnist_png/0/3.png")

inside = im > 0
outside = im <= 0

im1 = ndimage.morphology.distance_transform_edt(inside)
im2 = ndimage.morphology.distance_transform_edt(outside)

imf = im2-im1
imf = (imf - np.min(imf))/np.ptp(imf)

imn = -im

fig = plt.figure(figsize=(9,3))

ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(im, cmap="Greys")
plt.title("Original scalar function")
plt.axis('off')

ax = fig.add_subplot(1, 3, 2)
plt.title("Negated Scalar function")
imgplot = plt.imshow(imn, cmap="Greys")
plt.axis('off')

ax = fig.add_subplot(1, 3, 3)
plt.title("Distance transform")
imgplot = plt.imshow(imf, cmap="Greys")
plt.axis('off')

fig.savefig('../pictures/filtrations.png')