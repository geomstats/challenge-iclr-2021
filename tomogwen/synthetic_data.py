
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import random


def plot_all(data, diagrams):

    fig = plt.figure(figsize=(20, 10))

    for i in range(len(data)):
        num = 241 + i

        ax = plt.subplot(num)
        plt.scatter(data[i][:, 0], data[i][:, 1])

        ax = plt.subplot(num + 4)
        plot_diagram(diagrams[i], ax, lims=[0, 1.5, 0, 1.75])

    fig.suptitle("Datasets with corresponding persistence diagrams")
    plt.show()


def compute_diagrams(data):

    diagrams = []
    for i in range(len(data)):
        print("Processing data: " + str(i))
        filtration = dion.fill_rips(data[i], 2, 3.0)
        homology = dion.homology_persistence(filtration)
        diagram = dion.init_diagrams(homology, filtration)
        diagrams.append(diagram[1])
    print()

    return diagrams


def plot_clusters(M):
    plt.scatter(M[0].T[0], M[0].T[1], c='r', label='Rings')
    plt.scatter(M[1].T[0], M[1].T[1], c='b', label='Noise')
    plt.xlim([0, 1.5])
    plt.ylim([0, 1.75])
    plt.plot([0.1, 1.2], [0.1, 1.2])
    plt.legend()
    plt.title("Persistence Diagram Cluster Centres")

    plt.show()


def gen_data(seed, noise=0.05, n_samples=100):

    # print("\nGenerating data...\n")

    np.random.seed(seed)
    random.seed(seed)

    data = []
    data.append(datasets.make_circles(n_samples=n_samples, factor=0.99, noise=noise, random_state=seed)[0])
    data.append(datasets.make_circles(n_samples=n_samples, factor=0.99, noise=noise, random_state=seed + 1)[0])
    data.append(np.random.normal(size=(100, 2), scale=0.5))
    data.append(0.9 * np.random.normal(size=(100, 2), scale=0.5))

    return data


def gen_data2(seed, noise, n_samples):
    dataset = []

    np.random.seed(seed)
    random.seed(seed)

    # Noise
    data = np.random.normal(size=(100, 2), scale=0.5)
    dataset.append(data)

    data = np.random.normal(size=(100, 2), scale=0.5)
    data[:, 0] = data[:, 0] * 0.5
    dataset.append(data)

    data = np.random.normal(size=(100, 2), scale=0.5)
    data[:, 1] = data[:, 1] * 0.7
    dataset.append(data)

    # One Ring (to rule them all)
    data = datasets.make_circles(n_samples=n_samples, factor=0.99, noise=noise, random_state=seed)[0]
    dataset.append(data)

    data = datasets.make_circles(n_samples=n_samples, factor=0.99, noise=noise, random_state=seed+1)[0]
    data[:, 0] = data[:, 0] * 0.5
    dataset.append(data)

    data = datasets.make_circles(n_samples=n_samples, factor=0.99, noise=noise * 1.5, random_state=seed+2)[0]
    dataset.append(data)

    # Two Rings
    data1 = datasets.make_circles(n_samples=int(0.5*n_samples), factor=0.99, noise=noise, random_state=seed + 3)[0]
    data1[:, 1] -= 1
    data2 = datasets.make_circles(n_samples=int(0.5*n_samples), factor=0.99, noise=noise, random_state=seed + 4)[0]
    data2[:, 1] += 1
    data = np.concatenate((0.5 * data1, 0.5 * data2), axis=0)
    dataset.append(data)

    data1 = datasets.make_circles(n_samples=int(0.5*n_samples), factor=0.99, noise=noise, random_state=seed + 5)[0]
    data1[:, 1] -= 1
    data2 = datasets.make_circles(n_samples=int(0.5*n_samples), factor=0.99, noise=noise, random_state=seed + 6)[0]
    data2[:, 1] += 1
    data = np.concatenate((0.5 * data1, 0.5 * data2), axis=0)
    data = np.rot90(data).T
    dataset.append(data)

    data1 = datasets.make_circles(n_samples=int(0.5*n_samples), factor=0.99, noise=noise, random_state=seed+7)[0]
    data1[:, 1] -= 1
    data2 = datasets.make_circles(n_samples=int(0.5*n_samples), factor=0.99, noise=noise*2, random_state=seed+8)[0]
    data2[:, 1] += 1
    data = np.concatenate((0.5*data1, 0.5*data2), axis=0)
    dataset.append(data)

    return dataset


def plot_dataset(dataset):
    fig = plt.figure(figsize=(10, 10))
    lim = 1.45

    for i in range(len(dataset)):
        num = 331 + i
        ax = plt.subplot(num)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.scatter(dataset[i][:, 0], dataset[i][:, 1])

    plt.show()


def plot_everything(dataset, diagrams):
    fig = plt.figure(figsize=(20, 10))
    lim = 1.45

    for i in range(3):
        num = i+1
        ax = plt.subplot(3, 6, num)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.scatter(dataset[i][:, 0], dataset[i][:, 1])

        ax = plt.subplot(3, 6, num+3)
        plot_diagram(diagrams[i], ax, lims=[0, 1.5, 0, 1.75])

    for i in range(3):
        num = 7+i
        ax = plt.subplot(3, 6, num)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.scatter(dataset[i+3][:, 0], dataset[i+6][:, 1])

        ax = plt.subplot(3, 6, num+3)
        plot_diagram(diagrams[i+3], ax, lims=[0, 1.5, 0, 1.75])

    for i in range(3):
        num = 13+i
        ax = plt.subplot(3, 6, num)
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.scatter(dataset[i+6][:, 0], dataset[i+6][:, 1])

        ax = plt.subplot(3, 6, num+3)
        plot_diagram(diagrams[i+6], ax, lims=[0, 1.5, 0, 1.75])

    plt.show()


def plot_all_diagrams(diagrams):
    fig = plt.figure(figsize=(10, 10))

    for i in range(len(diagrams)):
        num = 331 + i

        ax = plt.subplot(num)
        plot_diagram(diagrams[i], ax, lims=[0, 1.5, 0, 1.75])

    # fig.suptitle("Datasets with corresponding persistence diagrams")
    plt.show()


def plot_diagram(dgm, ax, show=False, labels=False, line_style=None, pt_style=None, lims=False):

    # taken from Dionysus2 package

    line_kwargs = {}
    pt_kwargs = {}
    if pt_style is not None:
        pt_kwargs.update(pt_style)
    if line_style is not None:
        line_kwargs.update(line_style)

    inf = float('inf')

    if lims==False:
        min_birth = min(p.birth for p in dgm if p.birth != inf)
        max_birth = max(p.birth for p in dgm if p.birth != inf)
        min_death = min(p.death for p in dgm if p.death != inf)
        max_death = max(p.death for p in dgm if p.death != inf)

    else:
        min_birth = lims[0]
        max_birth = lims[1]
        min_death = lims[2]
        max_death = lims[3]

    ax.set_aspect('equal', 'datalim')

    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)
    ax.scatter([p.birth for p in dgm], [p.death for p in dgm], **pt_kwargs, color='g')
    ax.plot([min_diag, max_diag], [min_diag, max_diag], **line_kwargs)

    # ax.set_xlabel('birth')
    # ax.set_ylabel('death')
    ax.set_xticks([])
    ax.set_yticks([])


def plot_three_clusters(M):

    fig = plt.figure(figsize=(3.33, 10))

    lims = [0, 1.5, 0, 1.75]
    min_birth = lims[0]
    max_birth = lims[1]
    min_death = lims[2]
    max_death = lims[3]

    # diagram 1
    ax = plt.subplot(313)
    ax.set_aspect('equal', 'datalim')
    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)

    ax.scatter(M[0][:, 0], M[0][:, 1], color='g')
    ax.plot([min_diag, max_diag], [min_diag, max_diag])

    ax.set_xticks([])
    ax.set_yticks([])

    # diagram 2
    ax = plt.subplot(311)
    ax.set_aspect('equal', 'datalim')
    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)

    ax.scatter(M[1][:, 0], M[1][:, 1], color='g')
    ax.plot([min_diag, max_diag], [min_diag, max_diag])

    ax.set_xticks([])
    ax.set_yticks([])

    # diagram 3
    ax = plt.subplot(312)
    ax.set_aspect('equal', 'datalim')
    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)

    ax.scatter(M[2][:, 0], M[2][:, 1], color='g')
    ax.plot([min_diag, max_diag], [min_diag, max_diag])

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


if __name__ == '__main__':

    seed = 0
    dataset = gen_data2(seed, noise=0.1, n_samples=100)

    diagrams = compute_diagrams(dataset)
    diagrams_cluster = clustering.reformat_diagrams(diagrams)
    r, M = clustering.pd_fuzzy(diagrams_cluster, 3, verbose=True, max_iter=20)

    print("Membership values")
    print(r)

    plot_dataset(dataset)
    plot_all_diagrams(diagrams)
    plot_three_clusters(M)

    # Other synthetic data, not used in the paper
    # data = gen_data(seed, noise=0.3)
    # plot_all(data, diagrams)
    # plot_clusters(M)
    # plot_everything(dataset, diagrams)
