import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

fig = plt.figure()
number = 0
for subdir in os.listdir('../data/mnist_png'):
    if not subdir.startswith('.'):
        counter = 1 + number;
        for file in os.listdir(os.path.join('../data/mnist_png',subdir)):
            if not file.startswith('.'):
                ax = fig.add_subplot(6, 10, counter)
                counter += 10
                img = mpimg.imread(os.path.join('../data/mnist_png',subdir,file))
                imgplot = plt.imshow(img, cmap="Greys")
                plt.axis('off')
                if(counter > 60):
                    break
        number += 1

fig.savefig('../pictures/dataset.png')