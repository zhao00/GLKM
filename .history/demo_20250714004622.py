import pandas as pd
from GKM import GKM
from LKM.FLK import FLK
import os
import time
import csv
import numpy as np

from FeiPub import Gfuns, Mfuns, IFuns




data_path = './dataset/'
data_name = 'Mnist10k'  
X, y_true, N, dim, k = IFuns.loadmat("{}{}.mat".format(data_path,data_name))

knn = 100
NN, NND = Gfuns.knn_nn(X, min(knn, N-1), squared = True)

obj = GKM(NN=NN.astype(np.int32), NND=NND.astype(np.float64), c_true=k, debug=0)
obj.opt()
y_pred0 = obj.y_pred

CD  = False
if CD:
    ## Following is the coordinate descent fine-tuning stage of the objective function, which is optional.
    if not obj.connected:  
        print('The k-nearest neighbor graph is not connected. Recommend increase the value of k.')   
    y = IFuns.replace_with_continuous(y_pred0)
    init_Y = [y]
    mod = FLK(NN=NN.astype(np.int32), NND=NND.astype(np.float64), max_d=-1, c_true=k, debug=False)
    mod.opt(ITER=100, init_Y=init_Y)
    y_pred = mod.Y[0]
else:
    y_pred = y_pred0

acc = Mfuns.accuracy(y_true, y_pred)
nmi = 	Mfuns.nmi(y_true, y_pred)
purity = Mfuns.purity(y_true, y_pred)
ari = Mfuns.ari(y_true, y_pred)
print(f'{data_name}: \tacc={acc:.3f}, nmi={nmi:.3f}, purity={purity:.3f}')


