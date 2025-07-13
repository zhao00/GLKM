import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from FeiPub import Ifuns, Mfuns, Gfuns
sys.path.append("/home/pei/FLK/FLK_code")
from FLK import FLK
from FeiPub import Ifuns, Mfuns, Gfuns
from sklearn.neighbors import BallTree
import matplotlib.transforms as mt


def get_neighbors(X, k):
    tree = BallTree(X, leaf_size=2)              
    dist, ind = tree.query(X, k=k) 
    dist = dist.astype(np.float64)
    ind = ind.astype(np.int32)
    return ind, dist


def gen_ellipse(cen, w, h, N):
    X = np.zeros((N, 2))
    theta = np.linspace(0, 2*math.pi, N + 1)
    for i, t in enumerate(theta[:-1]):
        X[i, 0] = w * math.cos(t)
        X[i, 1] = h * math.sin(t)
    X += cen
    return X


def gen_mickey():
    T1 = gen_ellipse(np.array([-3, 5]), 1, 1, 10)
    T2 = gen_ellipse(np.array([-3, 5]), 0.5, 0.5, 3)

    X1 = np.concatenate((T1, T2), axis=0)
    y1 = np.ones(X1.shape[0]) * 0

    T1 = gen_ellipse(np.array([3, 5]), 1, 1, 10)
    T2 = gen_ellipse(np.array([3, 5]), 0.5, 0.5, 3)
    X2 = np.concatenate((T1, T2), axis=0)
    y2 = np.ones(X2.shape[0]) * 1

    T1 = gen_ellipse(np.array([0, 0]), 3.5, 3.5, 14)
    T2 = gen_ellipse(np.array([0, 0]), 2.3, 2.3, 8)
    T3 = gen_ellipse(np.array([0, 0]), 1, 1, 3)
    X3 = np.concatenate((T1, T2, T3), axis=0)
    y3 = np.ones(X3.shape[0]) * 2

    X = np.concatenate((X1, X2, X3), axis=0)
    y = np.concatenate((y1, y2, y3), axis=0)
    return X, y


if __name__ == "__main__":

    X, y = gen_mickey()

    knn = 10
    NN, NND = get_neighbors(X, knn)
    mod = FLK(NN=NN, NND = NND, c_true=3, debug=False, max_d=-1)
    init_Y = Ifuns.initialY("random", X.shape[0], 3, 1)
    mod.opt(ITER=100, init_Y=init_Y)
    Y = mod.Y

    y = np.array(["yexxxxllow"] * X.shape[0])
    y[Y[0] == 0] = "#e1701a"
    y[Y[0] == 1] = "#4ca1a3"
    y[Y[0] == 2] = "#344fa1"
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=72)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("./fig/mickey_flk.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()

    y_pred = KMeans(3).fit(X).labels_
    y = np.array(["yexxxxllow"] * X.shape[0])
    y[y_pred == 0] = "#e1701a"
    y[y_pred == 1] = "#4ca1a3"
    y[y_pred == 2] = "#344fa1"
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=72)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("./fig/mickey_km.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()