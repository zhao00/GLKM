import numpy as np
import sys
import matplotlib.pyplot as plt

from FLK import FLK
from FeiPub import Ifuns, Gfuns, Mfuns

X, y_true, N, dim, c_true = Ifuns.load_mat("./data/CASIA_FaceV5_29v2_20200916.mat")
knn = 20
NN, NND = Gfuns.knn_f(X, knn)
init_Y = Ifuns.initialY("random", N, c_true, 1)
print(init_Y)
print(N, dim, c_true)
print(NN)

print("Local KMeans (LKM)")
mod = FLK(NN=NN.astype(np.int32), NND=NND.astype(np.float64), max_d=-1, c_true=c_true, debug=False)
mod.opt(ITER=100, init_Y=init_Y)
acc = Mfuns.multi_accuracy(y_true, mod.Y)
nmi = Mfuns.multi_nmi(y_true, mod.Y)
ari = Mfuns.multi_ari(y_true, mod.Y)
print(acc, nmi, ari)

print("Fast Spectral Clustering (FSC)")
t = np.mean(NND)
Val = -np.exp(-NND / (2 * t ** 2))
Val[:, 0] = -np.sum(Val[:, 1:], axis=1)

mod = FLK(NN=NN.astype(np.int32), NND=Val.astype(np.float64), max_d=0, c_true=c_true, debug=False)
mod.opt(ITER=100, init_Y=init_Y)
acc = Mfuns.multi_accuracy(y_true, mod.Y)
nmi = Mfuns.multi_nmi(y_true, mod.Y)
ari = Mfuns.multi_ari(y_true, mod.Y)
print(acc, nmi, ari)