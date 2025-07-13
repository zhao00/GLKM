import pandas as pd
from GKM import GKM
from LKM.FLK import FLK
import os
import time
import csv
import numpy as np

from FeiPub import Gfuns, Mfuns, IFuns
# from FeiPub import Funs as Ifuns

data_name = 'USPS'  ## 
data_path = './dataset/'


X, y_true, N, dim, k = IFuns.loadmat("{}{}.mat".format(data_path,data_name))

knn = 20
NN, NND = Gfuns.knn_nn(X, min(knn, N-1), squared = True)    # 采用sklearn快速构建近邻图  


obj = GKM(NN=NN.astype(np.int32), NND=NND.astype(np.float64), k, debug=0)
obj.opt()
y_pred0 = obj.y_pred
if not obj.connected:   ## 如果不连通，就没有写入csv文件，应该改为写入一个特定值
    print('not connect')
y = IFuns.replace_with_continuous(y_pred0)
init_Y = [y]
mod = FLK(NN=NN.astype(np.int32), NND=NND.astype(np.float64), max_d=-1, c_true=k, debug=False)
mod.opt(ITER=100, init_Y=init_Y)
t4 = time.time()
y_pred = mod.Y[0]

acc = Mfuns.accuracy(y_true, y_pred)
nmi = 	Mfuns.nmi(y_true, y_pred)
purity = Mfuns.purity(y_true, y_pred)
ari = Mfuns.ari(y_true, y_pred)
print(f'{data_name}: \tacc={acc:.3f}, nmi={nmi:.3f}, purity={purity:.3f}')


