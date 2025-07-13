import os
import sys
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
import matplotlib.transforms as mt
import matplotlib as mpl


sys.path.append("/home/pei/FLK/FLK_code")
from FLK import FLK
from FeiPub import Ifuns, Mfuns, Gfuns


def get_neighbors(X, k):

    tree = BallTree(X, leaf_size=2)              
    dist, ind = tree.query(X, k=k) 
    dist = dist.astype(np.float64)
    ind = ind.astype(np.int32)
    return ind, dist


def gen_clusters(centers, cov, N):
    """
    return X (list), y (list)
    """
    X = list()
    y = list()
    cou = 0
    for cen in centers:
        X_tmp = np.random.multivariate_normal(mean=cen, cov=cov, size=N)
        X.append(X_tmp)

        y_tmp = [cou] * X_tmp.shape[0]
        y.extend(y_tmp)
        cou += 1
    return X, y


def gen_grid(c_true, sigma=0.2, N=10, dim=2):

    max_side = np.sqrt(c_true)
    side = np.arange(1, max_side+1)
    tmp = itertools.product(side, side)
    centers = np.array(list(tmp))

    cov = np.array([
        [sigma, 0],
        [0, sigma]
    ])

    X, y = gen_clusters(centers, cov, N)

    X2 = np.concatenate(X, axis=0)
    y2 = np.array(y, dtype=np.int32)

    if dim > 2:
        Noise = np.random.rand(X2.shape[0], dim - 2) / 5
        X2 = np.concatenate((X2, Noise), axis=1)
    return X2, y2


def get_objective_kmeans(X, y, c_true):
    Cen = np.zeros((c_true, X.shape[1]))
    n = np.zeros(c_true)
    for i, yi in enumerate(y):
        Cen[yi] += X[i]
        n[yi] += 1

    for i in range(c_true):
        Cen[i] /= n[i]

    obj = np.linalg.norm(X - Cen[y, :], 'fro') ** 2
    return obj


def print_result(X, y_true, Y, t, Iters):
    c_true = len(np.unique(y_true))
    pre = Mfuns.multi_precision(y_true, Y)
    rec = Mfuns.multi_recall(y_true, Y)
    f1 = 2 * pre * rec / (pre + rec)
    obj = np.array([get_objective_kmeans(X, y, c_true) for y in Y])

    pre = np.mean(pre)
    rec = np.mean(rec)
    f1 = np.mean(f1)
    obj = np.mean(obj)

    print(f"pre = {pre:.3f}, rec = {rec:.3f}, f1 = {f1:.3f}, obj = {obj:.2f}, time={t:.5f}, iter={np.mean(Iters)}")


if __name__ == "__main__":

    from cycler import cycler
    mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')

    # plot
    dim = 2
    sigma = 0.5
    c_true = 196
    max_iter = 100
    # f_name = f"/home/pei/FLK/FLK_code/data/grid_{c_true}_{sigma:.2f}_{dim}.mat"

    X, y_true = gen_grid(c_true, sigma=(sigma/3)**2, N=10, dim=dim)
    # data = sio.loadmat(f_name)
    # X = data["X"]
    # y_true = data["y_true"].reshape(-1)

    #  KMeans
    mod = KMeans(c_true, init="random", n_init=1, algorithm="full", max_iter=max_iter).fit(X)
    y_km = mod.labels_
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:, 0], X[:, 1], c=y_km, s=12)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("/home/pei/FLK/FLK_code/fig/grid_km.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()

    # FLK
    knn = 20
    NN, NND = get_neighbors(X, knn)
    mod = FLK(NN=NN, NND = NND, c_true=c_true, debug=False, max_d=-1)
    init_Y = Ifuns.initialY("random", X.shape[0], c_true, 1)
    mod.opt(ITER=max_iter, init_Y=init_Y)
    y_lkm = mod.Y[0]


    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:, 0], X[:, 1], c=y_lkm, s=12)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("/home/pei/FLK/FLK_code/fig/grid_lkm.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()

    dim = 2
    ktimes = 50
    max_iter = 100
    for c_true in [196, 3136, 12544]:
        for sigma in [0.5, 0.6, 0.7]:
            print("===================")
            print(f"c_true = {c_true}, sigma = {sigma}")

            # f_name = f"/home/pei/FLK/FLK_code/data/grid_{c_true}_{sigma:.2f}_{dim}.mat"

            X, y_true = gen_grid(c_true, sigma=(sigma/3)**2, N=10, dim=dim)
            #  Ifuns.save_mat(f_name, {"X": X, "y_true": y_true})

            # data = sio.loadmat(f_name)
            # X = data["X"]
            # y_true = data["y_true"].reshape(-1)

            #  KMeans
            t1 = time.time()
            Y = np.zeros((ktimes, X.shape[0]), dtype=np.int32)
            Iters = np.zeros(ktimes)
            for i in range(ktimes):
                mod = KMeans(c_true, init="random", n_init=1, algorithm="full", max_iter=max_iter).fit(X)
                Iters[i] = mod.n_iter_
                Y[i] = mod.labels_
            t2 = time.time()
            t = (t2 - t1) / ktimes
            print_result(X, y_true, Y, t, Iters)

            # FLK
            knn = 20
            t1 = time.time()
            NN, NND = get_neighbors(X, knn)
            t2 = time.time()
            print(f"knn time = {t2 - t1:.5f}")

            t1 = time.time()
            mod = FLK(NN=NN, NND = NND, c_true=c_true, debug=False, max_d=-1)
            init_Y = Ifuns.initialY("random", X.shape[0], c_true, ktimes)
            mod.opt(ITER=max_iter, init_Y=init_Y)
            t2 = time.time()
            Y = mod.Y
            Iters = mod.n_iter
            t = (t2 - t1) / ktimes
            print_result(X, y_true, Y, t, Iters)
