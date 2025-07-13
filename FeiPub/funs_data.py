import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.transforms as mt


def loadmat(path, to_dense=True):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["y_true"].astype(np.int32).reshape(-1)

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true

def four_gaussian(N=50, width=10, heigh=10):
    X1 = turdata(N, tur=0.3, d=2) + np.array([width/2, heigh/2])
    X2 = turdata(N, tur=0.3, d=2) + np.array([-width/2, heigh/2])
    X3 = turdata(N, tur=0.3, d=2) + np.array([width/2, -heigh/2])
    X4 = turdata(N, tur=0.3, d=2) + np.array([-width/2, -heigh/2])

    X = np.concatenate((X1, X2, X3, X4), axis=0)
    c_true = 4
    y = np.repeat(np.arange(c_true), repeats=N)
    num, dim = X.shape
    return X, y, num, dim, c_true

def rand_ring(r, N, tur, cx=0, cy=0, s=0, e=2 * np.pi, dis="gaussian"):
    """
    dis=["uniform", "gaussian"]
    """
    theta = np.linspace(s, e, N)
    x = np.vstack((r * np.cos(theta), r * np.sin(theta))).T
    x = x + turdata(N, tur, d=2, dis=dis)

    x[:, 0] += cx
    x[:, 1] += cy

    return x

# sub function
def turdata(N, tur, d=2, dis="gaussian"):
    """
    dis=["uniform", "gaussian"]
    """

    if dis == "gaussian":
        mu = np.repeat(0, d)
        sig = np.eye(d) * tur
        x = np.random.multivariate_normal(mu, sig, N)
    elif dis == "uniform":
        x = np.random.uniform(-tur/2, tur/2, (N, 2))
    return x


def twospirals(N=2000, degrees=570, start=90, noise=0.2, seed=0):
    np.random.seed(seed)

    X = np.zeros((N, 2), dtype=np.float64)
    deg2rad = np.pi/180
    start = start * deg2rad

    N1 = int(np.floor(N/2))
    N2 = N - N1

    n = start + np.sqrt(np.random.rand(N1)) * degrees * deg2rad
    X[:N1, 0] = -np.cos(n) * n + np.random.rand(N1) * noise
    X[:N1, 1] =  np.sin(n) * n + np.random.rand(N1) * noise

    n = start + np.sqrt(np.random.rand(N2)) * degrees * deg2rad
    X[N1:, 0] =  np.cos(n) * n + np.random.rand(N2) * noise
    X[N1:, 1] = -np.sin(n) * n + np.random.rand(N2) * noise

    y = np.ones(N, dtype=np.int32)
    y[:N1] = 0
    return X, y

def rings(c_true, num_coe=1, r_coe=1, width=0.2, seed=0):
    r_arr = np.arange(1, c_true+1) * r_coe
    s = 2 * np.pi * r_arr

    N = int(100 * num_coe)
    N_arr = (s/s[0] * N).astype(np.int32)

    
    X_list = list()
    y_list = list()
    for i in range(c_true):
        r = r_arr[i]
        N = N_arr[i]

        phi = np.random.rand(N) * 2 * np.pi
        dist = r + (np.random.rand(N) - 0.5) * width
        x0 = dist * np.cos(phi)
        x1 = dist * np.sin(phi)
        Xi = np.concatenate((x0.reshape(-1, 1), x1.reshape(-1, 1)), axis=1)
        X_list.append(Xi)

        y = np.zeros(N) + i
        y_list.append(y)
    
    X = np.concatenate(X_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)
    return X, y_true

def corners(num_coe=1, width=1, gap=2, l=5, seed=0):
    assert l > width

    np.random.seed(seed)
    N = int(200 * num_coe)

    #   (x0, y2)         (x1, y2)
    # 
    #
    #   (x0, y1)         (x1, y1)
    #    
    #   (x0, y0) -width- (x1, y0) ................ (x2, y0)

    x0, y0 = gap/2, gap/2
    x1, y1 = x0 + width, y0 + width
    x2, y2 = x0 + l, y0 + l

    N1_ratio = (l-width) / (2 * l - width)
    N1 = int(N * N1_ratio)
    N2 = N - N1
    X1_part1 = np.random.uniform([x0, y1], [x1, y2], [N1, 2])
    X1_part2 = np.random.uniform([x0, y0], [x2, y1], [N2, 2])
    X1 = np.concatenate((X1_part1, X1_part2), axis=0)

    X2_part1 = np.random.uniform([-x1, y1], [-x0, y2], [N1, 2])
    X2_part2 = np.random.uniform([-x2, y0], [-x0, y1], [N2, 2])
    X2 = np.concatenate((X2_part1, X2_part2), axis=0)

    X3_part1 = np.random.uniform([-x1, -y2], [-x0, -y1], [N1, 2])
    X3_part2 = np.random.uniform([-x2, -y1], [-x0, -y0], [N2, 2])
    X3 = np.concatenate((X3_part1, X3_part2), axis=0)

    X4_part1 = np.random.uniform([x0, -y2], [x1, -y1], [N1, 2])
    X4_part2 = np.random.uniform([x0, -y1], [x2, -y0], [N2, 2])
    X4 = np.concatenate((X4_part1, X4_part2), axis=0)

    X = np.concatenate((X1, X2, X3, X4), axis=0)
    y_true = np.repeat(np.arange(4), N)
    return X, y_true

def data_description(data_path, data_name, version, url):
    full_name = os.path.join(data_path, f"{data_name}_{version}.mat")
    X, y_true, N, dim, c_true = loadmat(full_name)

    # title and content
    T1 = "data_name"
    T2 = "# Samples"
    T3 = "# Features"
    T4 = "# Subjects"

    C1 = data_name
    C2 = str(X.shape[0])
    C3 = str(X.shape[1])
    C4 = str(c_true)

    n1 = max(len(T1), len(C1))
    n2 = max(len(T2), len(C2))
    n3 = max(len(T3), len(C3))
    n4 = max(len(T4), len(C4))

    y_df = pd.DataFrame(data=y_true, columns=["label"])
    ind_L = y_df.groupby("label").size()

    show_n = 5

    with open("{}{}_{}.txt".format(data_path, data_name, version), "a") as f:

        # version
        f.write("version = {}\n\n".format(version))

        # table
        f.write("{}  {}  {}  {}\n".format(
            T1.rjust(n1), T2.rjust(n2), T3.rjust(n3), T4.rjust(n4)))
        f.write("{}  {}  {}  {}\n\n".format(
            C1.rjust(n1), C2.rjust(n2), C3.rjust(n3), C4.rjust(n4)))

        # url
        f.write("url = {}\n\n".format(url))
        f.write("=================================\n")

        # content
        f.write("X[:, :2], {}, {}, {}\n".format(
            str(type(X))[8:-2], X.shape, str(type(X[0, 0]))[8:-2]))
        if isinstance(X, sparse.spmatrix):
            f.write("{}\n".format(X[:show_n, :2].toarray()))
        else:
            f.write("{}\n".format(X[:show_n, :2]))
        f.write("...\n\n")

        f.write("y_true, {}, {}, {}\n".format(
            str(type(y_true))[8:-2], y_true.shape, str(type(y_true[0]))[8:-2]))
        f.write("{}".format(y_true[:show_n]))
        f.write("...\n\n")

        f.write("distribution\n")
        f.write(ind_L[:50].to_string())
        f.write("\n\n")

if  __name__ == "__main__":

    # X, y = twospirals()
    # X, y = rings(3, num_coe=2)
    X, y = corners(num_coe=1, width=1, gap=2, l=5, seed=0)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
