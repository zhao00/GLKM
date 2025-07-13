import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import matplotlib.transforms as mt


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


def gen_outliers(sigma=0.2, N=10):
    centers = np.array([
        [0, 0],
        [0, 5],
        [5, 0],
        [5, 5],
    ])

    cov = np.array([
        [sigma, 0],
        [0, sigma]
    ])

    X, y = gen_clusters(centers, cov, N)
    X.append(np.array([[100, 100]]))
    y.append(3)

    X2 = np.concatenate(X, axis=0)
    y2 = np.array(y, dtype=np.int32)

    return X2, y2


if __name__ == "__main__":

    np.random.seed(0)
    X, y_pred = gen_outliers(N=20)
    c_true = 4

    y = np.array(["yexxxxllow"] * X.shape[0])
    y[y_pred == 0] = "#e1701a"
    y[y_pred == 1] = "#4ca1a3"
    y[y_pred == 2] = "#344fa1"
    y[y_pred == 3] = "#c67ace"
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:-1, 0], X[:-1, 1], c=y[:-1], s=72)
    plt.axis("equal")
    plt.xlim([-1, 8])
    plt.ylim([-1, 8])
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("./fig/outliers_ori.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()

    knn = 10
    NN, NND = get_neighbors(X, knn)
    mod = FLK(NN=NN, NND = NND, c_true=c_true, debug=False, max_d=-1)
    init_Y = Ifuns.initialY("random", X.shape[0], c_true, 1)
    print(init_Y)
    mod.opt(ITER=100, init_Y=init_Y)
    y_pred = mod.Y[0]

    y = np.array(["yexxxxllow"] * X.shape[0])
    y[y_pred == 0] = "#e1701a"
    y[y_pred == 1] = "#4ca1a3"
    y[y_pred == 2] = "#344fa1"
    y[y_pred == 3] = "#c67ace"
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:-1, 0], X[:-1, 1], c=y[:-1], s=72)
    plt.axis("equal")
    plt.xlim([-1, 8])
    plt.ylim([-1, 8])
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("./fig/outliers_flk.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()

    y_pred = KMeans(c_true).fit(X).labels_
    y = np.array(["yexxxxllow"] * X.shape[0])
    y[y_pred == 0] = "#e1701a"
    y[y_pred == 1] = "#4ca1a3"
    y[y_pred == 2] = "#344fa1"
    y[y_pred == 3] = "#c67ace"
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(X[:-1, 0], X[:-1, 1], c=y[:-1], s=72)
    plt.axis("equal")
    plt.xlim([-1, 8])
    plt.ylim([-1, 8])
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0.05, 0.05)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("./fig/outliers_km.png", dpi = 300, bbox_inches=mt.Bbox([[-0.1, -0.1], [6.5, 4.9]]))
    plt.show()