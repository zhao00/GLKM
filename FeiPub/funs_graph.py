import time
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances as EuDist2
from sklearn.neighbors import NearestNeighbors
# from .CppFuns.CppFuns_ import symmetry_py
# from .CppFuns.CppFuns_ import knn_graph_tfree_py
from . import funs as Funs

def get_anchor(X, m, way="random"):
    """
    X: n x d,
    m: the number of anchor
    way: [k-means, k-means2, k-means++, k-means++2, random]
    """
    if way == "k-means":
        A = KMeans(m, init='random').fit(X).cluster_centers_
    elif way == "k-means2":
        A = KMeans(m, init='random').fit(X).cluster_centers_
        D = EuDist2(A, X)
        ind = np.argmin(D, axis=1)
        A = X[ind, :]
    elif way == "k-means++":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
    elif way == "k-means++2":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
        D = EuDist2(A, X)
        A = np.argmin(D, axis=1)
    elif way == "random":
        ids = random.sample(range(X.shape[0]), m)
        A = X[ids, :]
    else:
        raise SystemExit('no such options in "get_anchor"')
    return A



def knn_nn(X, knn, squared=True):

    # 计算K最近邻及其距离
    nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='auto', metric='euclidean')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    if squared:
        distances **= 2 # 距离平方
    
    return indices, distances

# 实际上，不需要截断第一列
'''
def knn_nn(X, knn, squared=True, include_self = False):   # 默认不包含自身
    # 初始化最近邻搜索对象
    
    nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm='auto', metric='euclidean')
    nbrs.fit(X)

    # 获取每个点的k+1个近邻
    distances, indices = nbrs.kneighbors(X)

    adjusted_distances = []
    adjusted_indices = []

    for i in range(len(X)):
        current_indices = indices[i]
        current_distances = distances[i]

        # 检查并排除自身索引
        mask = current_indices != i
        filtered_indices = current_indices[mask]
        filtered_distances = current_distances[mask]

        if not include_self:    # 如果不包含自身，恰好是k个
            # 只取前k个
            adjusted_indices.append(filtered_indices)
            adjusted_distances.append(filtered_distances)
        else:
            # 在第一个位置插入自身索引和距离0
            filtered_indices = np.insert(filtered_indices, 0, i)
            filtered_distances = np.insert(filtered_distances, 0, 0)
            # 只取前k个
            adjusted_indices.append(filtered_indices[:knn])
            adjusted_distances.append(filtered_distances[:knn])

    adjusted_distances = np.array(adjusted_distances)
    adjusted_indices = np.array(adjusted_indices)
    #return adjusted_indices, adjusted_distances

    if squared:
        adjusted_distances **= 2  # 距离平方


    return adjusted_indices, adjusted_distances  # 返回近邻索引和距离
'''


# 对于大数据集，下述暴力方法太慢了

def knn_f(X, knn, squared=True, self_include=True):
    t_start = time.time()

    D_full = EuDist2(X, X, squared=squared)
    np.fill_diagonal(D_full, -1)
    NN_full = np.argsort(D_full, axis=1)
    np.fill_diagonal(D_full, 0)

    if self_include:
        NN = NN_full[:, :knn]
    else:
        NN = NN_full[:, 1:(knn+1)]

    NND = Funs.matrix_index_take(D_full, NN)

    NN = NN.astype(np.int32)
    NND = NND.astype(np.float64)

    t_end = time.time()
    t = t_end - t_start

    return NN, NND, t


def build_symmetric_knn_graph_union(X, knn, squared=True):
    """
    构建对称KNN图。并集方式

    参数:
    X -- 样本矩阵，尺寸为 N x dim。
    knn -- 每个样本的最近邻数量。

    返回:
    neighbors_indices -- 对称KNN图的邻居索引列表。
    neighbors_distances -- 对称KNN图的邻居距离列表。
    """
    # 计算K最近邻及其距离
    nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='auto', metric='euclidean')
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:] # 排除每个点本身
    distances = distances[:, 1:]
    if squared:
        distances **= 2 # 距离平方
    
    max_distance = max(distances[:,-1])

    # 创建初始邻接列表
    N = X.shape[0]
    neighbors_indices = [[] for _ in range(N)]
    neighbors_distances = [[] for _ in range(N)]

    # 填充邻接列表（并集方式）
    for i in range(N):
        for j, dist in zip(indices[i], distances[i]):
            # 添加 i 的邻居 j
            if i != j:
                neighbors_indices[i].append(j)
                neighbors_distances[i].append(dist)
                # 同时检查并添加 j 的邻居 i
                if i not in indices[j]:
                    neighbors_indices[j].append(i)
                    neighbors_distances[j].append(dist)

    return neighbors_indices, neighbors_distances, max_distance

def knn_graph_gaussian(X, knn, t_way="mean", self_include=False, isSym=True):
    """
    :param X: data matrix of n by d
    :param knn: the number of nearest neighbors
    :param t_way: the bandwidth parameter
    :param self_include: weather xi is among the knn of xi
    :param isSym: True or False, isSym = True by default
    :return: A, a matrix (graph) of n by n
    """
    N = X.shape[0]
    NN, NND, time1 = knn_f(X, knn, squared=True, self_include=self_include)

    Val = dist2sim_kernel(NND, t_way=t_way)

    A = np.zeros((N, N))
    Funs.matrix_index_assign(A, NN, Val)
    np.fill_diagonal(A, 0)

    if isSym:
        A = (A + A.T) / 2

    return A

def dist2sim_kernel(NND, t_way="mean"):
    if t_way == "mean":
        t = np.mean(NND)
    elif t_way == "median":
        t = np.median(NND)
    else:
        raise SystemExit('no such options in "dist2sim_kernel, t_way"')

    Val = np.exp(-NND / (2 * t ** 2))

    return Val

def knn_graph_tfree(X, knn, self_include=False, isSym=True):
    """
    :param X: data matrix of n by d
    :param knn: the number of nearest neighbors
    :param self_include: weather xi is among the knn of xi
    :param isSym: True or False, isSym = True by default
    :return: A, a matrix (graph) of n by n
    """
    t_start = time.time()

    N = X.shape[0]
    NN_K, NND_K, time1 = knn_f(X, knn + 1, squared=True, self_include=self_include)

    NN = NN_K[:, :knn]
    NND = NND_K[:, :knn]
    NND_k = NND_K[:, knn]

    Val = dist2sim_t_free(NND, NND_k)

    A = np.zeros((N, N))
    Funs.matrix_index_assign(A, NN, Val)
    np.fill_diagonal(A, 0)

    if isSym:
        A = (A + A.T) / 2
    
    t_end = time.time()
    t = t_end - t_start
    return A, t

def dist2sim_t_free(NND, NND_k):
    knn = NND.shape[1]

    Val = NND_k.reshape(-1, 1) - NND

    Val[Val[:, 0] < 1e-6, :] = 1.0/knn

    Val = Val / (np.sum(Val, axis=1).reshape(-1, 1))
    return Val

def kng_anchor(X, Anchor: np.ndarray, knn=20, way="gaussian", t_way="mean", shape=None):
    """
    :param X: data matrix of n by d
    :param Anchor: Anchor set, m by d
    :param knn: the number of nearest neighbors
    :param way: one of ["gaussian", "t_free"]
        "t_free" denote the method proposed in : "The constrained laplacian rank algorithm for graph-based clustering"
        "gaussian" denote the heat kernel
    :param t_way: only needed by gaussian, the bandwidth parameter
    :return: A, a matrix (graph) of n by m
    """
    N = X.shape[0]
    anchor_num = Anchor.shape[0]

    # NN_K, NND_K
    D = EuDist2(X, Anchor, squared=True)  # n x m
    NN_full = np.argsort(D, axis=1)
    NN_K = NN_full[:, :(knn+1)]  # xi isn't among neighbors of xi
    NND_K = Funs.matrix_index_take(D, NN_K)

    # NN, NND, NND_k
    NN = NN_K[:, :knn]
    NND = NND_K[:, :knn]
    NND_k = NND_K[:, knn]

    if way=="gaussian":
        Val = dist2sim_kernel(NND, t_way=t_way)
    elif way=="t_free":
        Val = dist2sim_t_free(NND, NND_k)
    else:
        raise SystemExit('no such options in "get_anchor"')

    A = np.zeros((N, anchor_num))
    Funs.matrix_index_assign(A, NN, Val)
    return A
