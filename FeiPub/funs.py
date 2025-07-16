import os
import time
import random
import numpy as np
import pandas as pd
import scipy
import scipy.io as sio
from scipy import stats
from scipy import sparse
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
import pickle

def initial_Y(X, c, rep, way="random"):
    N = X.shape[0]
    Y = np.zeros((rep, N), dtype=np.int32)

    if way == "random":
        for rep_i in range(rep):
            Y[rep_i] = np.random.randint(0, c, N)
        
    elif way == "k-means++":
        for rep_i in range(rep):
            Y[rep_i] = KMeans(n_clusters=c, init="k-means++", n_init=1, max_iter=1).fit(X).labels_
    
    else:
        assert 2 == 1
    
    return Y

def replace_with_continuous(nums):
    num_dict = {}  # 创建一个空的字典用于存储数值映射关系
    new_nums = []  # 创建一个新的列表用于存储映射后的数值
    
    # 遍历列表中的每个数值
    for num in nums:
        # 如果这个数值不在字典中，将其映射到当前字典长度作为新的数值
        if num not in num_dict:
            num_dict[num] = len(num_dict)
        # 将映射后的数值添加到新列表中
        new_nums.append(num_dict[num])
    
    return new_nums


def savepkl(data, full_path, make_dir=True):

    if make_dir:
        path = os.path.dirname(full_path)
        if not os.path.exists(path):
            os.makedirs(path)

    pickle.dump(data, open(full_path, "wb"))

def loadpkl(full_path):
    data = pickle.load(open(full_path, "rb"))
    return data

def loadmat(path, to_dense=True):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["y_true"].astype(np.int32).reshape(-1)

    if sparse.isspmatrix(X) and to_dense:
        X = X.toarray()

    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    X = X.astype(np.float64)
    return X, y_true, N, dim, c_true

def load_toy(file_path):    ## grid dataset
    try:
        # 加载 .mat 文件
        data = sio.loadmat(file_path)
        # 提取需要的数据
        X = data["X"]
        y_true = data["y_true"].reshape(-1)
        NN = data["NN"]
        NND = data["NND"]
        knn_time = data["knn_time"][0][0]
        return X, y_true, NN, NND, knn_time
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except KeyError as e:
        print(f"数据集中缺少键 {e}。")
        return None


def savemat(full_path, xy, make_dir=True):
    if make_dir:
        path = os.path.dirname(full_path)
        if not os.path.exists(path):
            os.makedirs(path)

    sio.savemat(full_path, xy)

def matrix_index_take(X, ind_M):
    """
    :param X: ndarray
    :param ind_M: ndarray
    :return: X[ind_M] copied
    """
    assert np.all(ind_M >= 0)

    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    ret = X[row, col].reshape((n, k))
    return ret

def matrix_index_assign(X, ind_M, Val):
    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    if isinstance(Val, (float, int)):
        X[row, col] = Val
    else:
        X[row, col] = Val.reshape(-1)


def EProjSimplex_new(v, k=1):
    v = v.reshape(-1)
    # min  || x- v ||^2
    # s.t. x>=0, sum(x)=k
    ft = 1
    n = len(v)
    v0 = v-np.mean(v) + k/n
    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m -= f/g
            ft += 1
            if ft > 100:
                break
        x = np.maximum(v1, 0)
    else:
        x = v0

    return x, ft


def EProjSimplexdiag(d, u):
    #  d = d.astype(np.float64)
    #  u = u.astype(np.float64)
    # min  1/2*x'*U*x - x'*d
    # s.t. x>=0, sum(x) = 1
    lam = np.min(u-d)
    #  print(lam)
    f = 1
    count = 1
    while np.abs(f) > 1e-8:
        v1 = (lam + d)/u
        posidx = v1 > 0
        #  print(v1)
        g = np.sum(1/u[posidx])
        f = np.sum(v1[posidx]) - 1
        #  print(f)
        lam -= f/g

        if count > 1000:
            break
        count += 1
    v1 = (lam+d)/u
    x = np.maximum(v1, 0)
    return x, f

def eig1(A, c, isMax=True, isSym=True):
    if isinstance(A, sparse.spmatrix):
        A = A.toarray()

    if isSym:
        A = np.maximum(A, A.T)

    if isSym:
        d, v = np.linalg.eigh(A)
    else:
        d, v = np.linalg.eig(A)

    if isMax:
        idx = np.argsort(-d)
    else:
        idx = np.argsort(d)

    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]

    eigval_full = d[idx]

    return eigvec, eigval, eigval_full


def KMeans(X, c, rep, init="random", algorithm="auto"):
    '''
    :param X: 2D-array with size of N x dim
    :param c: the number of clusters to construct
    :param rep: the number of runs
    :param init: the way of initialization: random (default), k-means++
    :return: Y, 2D-array with size of rep x N, each row is a assignment
    '''
    times = np.zeros(rep)
    Y = np.zeros((rep, X.shape[0]), dtype=np.int32)
    for i in range(rep):
        t_start = time.time()
        Y[i] = sk_KMeans(n_clusters=c, n_init=1, init=init, algorithm=algorithm).fit(X).labels_
        t_end = time.time()
        times[i] = t_end - t_start

    return Y, times


def relabel(y, offset=0):
    y_df = pd.DataFrame(data=y, columns=["label"])
    ind_dict = y_df.groupby("label").indices

    for yi, ind in ind_dict.items():
        y[ind] = offset
        offset += 1
    return y


def normalize_fea(fea, row):
    '''
    if row == 1, normalize each row of fea to have unit norm;
    if row == 0, normalize each column of fea to have unit norm;
    '''

    if 'row' not in locals():
        row = 1

    if row == 1:
        feaNorm = np.maximum(1e-14, np.sum(fea ** 2, 1).reshape(-1, 1))
        fea = fea / np.sqrt(feaNorm)
    else:
        feaNorm = np.maximum(1e-14, np.sum(fea ** 2, 0))
        fea = fea / np.sqrt(feaNorm)

    return fea

def initialY(X, c_true, rep, way="random"):
    num, dim = X.shape
    if way == "random":
        Y = np.zeros((rep, num), dtype=np.int32)
        for i in range(rep):
            y1 = np.arange(c_true)      # 应该是避免出现少某个聚类的情况
            y2 = np.random.randint(0, c_true, num-c_true)
            y3 = np.concatenate((y1, y2), axis=0)
            np.random.shuffle(y3)      #随机打乱
            Y[i, :] = y3

    elif way == "k-means":
        Y, t = KMeans(X=X, c=c_true, rep=rep, init="random")
    elif way == "k-means++":
        Y, t = KMeans(X=X, c=c_true, rep=rep, init="k-means++")
    else:
        raise SystemExit('no such options in "initialY"')

    return Y

def y2Cen(X, y, c_true):
    eye_c = np.eye(c_true, dtype=np.int32)
    Y = eye_c[y]
    nc_neg = np.diag( 1 / np.diag(Y.T @ Y) )
    Cen = nc_neg @ (Y.T @ X)
    return Cen

def Y2Cens(X, Y, c_true):
    rep = Y.shape[0]
    dim = X.shape[1]
    Cen = np.zeros((rep, c_true, dim), dtype=np.float64)
    for i, y in enumerate(Y):
        Cen[i] = y2Cen(X, y, c_true)
    
    return Cen

def y2Y(y, c_true):
    eye_c = np.eye(c_true)
    Y = eye_c[y]
    return Y