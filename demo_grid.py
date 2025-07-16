from GKM import GKM
from LKM.FLK import FLK
import numpy as np
from FeiPub import Gfuns, Mfuns, IFuns


data_path = './dataset/'
data_name = 'D1'  
X, y_true, NN, NND, knn_time = IFuns.load_toy(f"{data_path}{data_name}.mat")
k = len(np.unique(y_true))      # c_true

print('----')

obj = GKM(NN=NN.astype(np.int32), NND=NND.astype(np.float64), c_true=k, debug=0)
obj.opt()
y_pred0 = obj.y_pred
print('----')

CD  = True      ## coordinate descent
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

## D1, acc=0.990, nmi=0.996, purity=0.990, (knn=n/c*1.2), time = 0.3373 + 3.9542
