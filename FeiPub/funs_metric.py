
import numpy as np
import pandas as pd
from scipy import stats

from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score as ari_ori
from sklearn.metrics import adjusted_mutual_info_score as ami_ori
from sklearn.metrics import normalized_mutual_info_score as nmi_ori
from sklearn.metrics import accuracy_score

from collections import Counter
# from . import Funs

def multi_kmeans_obj(X, Y):
    ret = np.array([kmeans_obj(X, y_pred) for y_pred in Y])
    return ret

# def kmeans_obj(X, y_pred, c_true):
#     c = len(np.unique(y_pred))
#     if c < c_true:
#         return -1
#     else:
#         Y = Funs.y2Y(y_pred, c_true)
#         nc_inv = np.diag(1/np.diag(Y.T @ Y))
#         cen = nc_inv @ Y.T @ X  # center = cxd
#         obj = np.linalg.norm(X - Y @ cen, "fro")
#         obj *= obj
#     return obj

def precision(y_true, y_pred):
    assert (len(y_pred) == len(y_true))
    N = len(y_pred)
    y_df = pd.DataFrame(data=y_pred, columns=["label"])
    ind_L = y_df.groupby("label").indices
    ni_L = [stats.mode(y_true[ind]).count[0] for yi, ind in ind_L.items()]
    return np.sum(ni_L) / N


def multi_precision(y_true, Y):
    ret = np.array([precision(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def recall(y_true, y_pred):
    re = precision(y_true=y_pred, y_pred=y_true)
    return re


def multi_recall(y_true, Y):
    ret = np.array([recall(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def accuracy(y_true, y_pred):
    """Get the best accuracy.

    Parameters
    ----------
    y_true: array-like
        The true row labels, given as external information
    y_pred: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    cost_m = np.max(cm) - cm
    indices = linear_sum_assignment(cost_m)
    indices = np.asarray(indices)
    indexes = np.transpose(indices)
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value
    return total * 1. / np.sum(cm)


def multi_accuracy(y_true, Y):
    ret = np.array([accuracy(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def fmi(y_true, y_pred):
    ret = metrics.fowlkes_mallows_score(labels_true=y_true, labels_pred=y_pred)
    return ret


def multi_fmi(y_true, Y):
    ret = np.array([fmi(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def ari(y_true, y_pred):
    ret = ari_ori(labels_true=y_true, labels_pred=y_pred)
    return ret


def multi_ari(y_true, Y):
    ret = np.array([ari(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def ami(y_true, y_pred, average_method="max"):
    ret = ami_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret


def multi_ami(y_true, Y):
    ret = np.array([ami(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def nmi(y_true, y_pred, average_method="max"):
    ret = nmi_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret


def multi_nmi(y_true, Y):
    ret = np.array([nmi(y_true=y_true, y_pred=y_pred) for y_pred in Y])
    return ret


def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

# 需要的是对称的NN, NND, 
def loss_element(label, NN, NND, max_distance):
    cluster_counts = Counter(label)

    # 将计数结果转换为一维数组，按照簇的编号顺序
    cluster_sizes = [cluster_counts[i] for i in range(max(label) + 1)]
    c_true = len(cluster_sizes)
    N = len(NN)
    count = [0] * c_true    # 每个类簇的计数
    objs = [0] * c_true

    # 是否应该采用对称的呢，采用对称会让loss变小
    #NN, NND, graph_time = Gfuns.knn_f(X, knn, self_include=False)
    for sample in range(N):  # 遍历每个样本i
        for index, j in enumerate(NN[sample]):  # 挨个遍历sample的每个近邻j
            if label[j] == label[sample]:   # 不同类的不计算
                objs[label[sample]] += NND[sample][index]       # 还涉及到对称性的问题
                count[label[sample]] += 1   # count记录簇内已经统计了的距离个数

    # obj = [x / y for x, y in zip(objs, cluster_sizes)]
    # obj_GKM = sum(obj)

    combinations = [size * (size - 1)  for size in cluster_sizes]
    # 然后，我们计算 diff，即 combinations 减去 count
    diff = [comb - c for comb, c in zip(combinations, count)]

    others = [a * max_distance for a in diff]   # 每个簇内需要补充的距离
    obj_GKM = [a + b for a, b in zip(objs, others)]  # 每个类簇内总共的距离

    single = [x / y for x, y in zip(obj_GKM, cluster_sizes)]
    return sum(single)

def loss_GKM(label, NN, NND):   ### 注意，采用对称的KNN图，

    max_distance = 0
    for i in range(len(NN)):
        max_distance = max(max_distance,NND[i][-1])

    # 下面进行对称操作
    N = len(label)
    neighbors_indices = [[] for _ in range(N)]
    neighbors_distances = [[] for _ in range(N)]

    # 填充邻接列表（并集方式）
    for i in range(N):
        for j, dist in zip(NN[i], NND[i]):
            # 添加 i 的邻居 j
            if i != j:
                neighbors_indices[i].append(j)
                neighbors_distances[i].append(dist)
                # 同时检查并添加 j 的邻居 i
                if i not in NN[j]:
                    neighbors_indices[j].append(i)
                    neighbors_distances[j].append(dist)

    return loss_element(label, neighbors_indices, neighbors_distances, max_distance)
    # Y = build_indicator_matrix(label)
    # D_clipped = construct_full_distance_matrix(NN, NND)
    # loss_matrix = loss(Y, D_clipped)
    # return loss_matrix

'''
def build_indicator_matrix(labels):
    # 创建一个N×K的零矩阵
    num_clusters = len(np.unique(labels))
    N = len(labels)
    Y = [[0 for _ in range(num_clusters)] for _ in range(N)]
    
    # 填充矩阵
    for i, label in enumerate(labels):
        Y[i][label] = 1
    
    Y = np.array(Y)
    return Y
def loss(Y, D):
    #loss = np.trace(Y.T@D@Y)
    loss = np.trace(np.linalg.inv(Y.T@Y)@Y.T@D@Y)
    return loss

def construct_full_distance_matrix(NN, NND):
    # Number of points
    n_points = NN.shape[0]
    max_distance = max(NND[:,-1])
    
    # Initialize the full distance matrix with infinities
    D_full = np.full((n_points, n_points), max_distance)
    
    for i in range(n_points):   # 对角线元素应为0
        D_full[i][i] = 0

    # Fill the distance matrix with the nearest neighbor distances
    for i in range(n_points):
        D_full[i, NN[i]] = NND[i]
        D_full[NN[i], i] = NND[i]
    
    # # Symmetrize the distance matrix
    # i_lower = np.tril_indices(n_points, -1)
    # D_full[i_lower] = D_full.T[i_lower]
    
    return D_full
'''