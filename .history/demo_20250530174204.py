import pandas as pd
import os
import csv
import sys
import time
from GKM import GKM
from LKM.FLK import FLK
import numpy as np
from memory_profiler import profile
import psutil, os
import traceback
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # 计算上级目录的路径
# sys.path.insert(0, parent_dir)      # 将上级目录添加到 sys.path
from FeiPub import IFuns as Ifuns
from FeiPub  import Gfuns, Mfuns

data_path  = 'D:\\DATA\\'
dataset_sorted = pd.read_csv(data_path+'dataset_sorted.csv',delimiter='\t',
                             dtype={'data_name':str,'number of sample':int,'feature':int,'c_true':int})

# dataset_list = ['Olivetti','Umist','Coil20','Palm','CASIA_FaceV5_29v2','Face94_256','CMUPIE','optdigits','Gisette','USPS','Letter','Mnist']
dataset_list = ['Iris', 'Aloi', 'CACD_29v2', 'IMDB_29v2','CASIA_WebFace_29v2']
dataset_list = ['Iris','IMDB_29v2',]
dataset_list = ['USPS']
print(dataset_list)

knn_list = {'Iris':40, 'Aloi':100, 'CACD_29v2':100, 'IMDB_29v2':1000, 'CASIA_WebFace_29v2':1000, 'Palm':20, 'Mnist':100, 'USPS':50}
def snapshot_memory(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[MEMORY SNAPSHOT] {tag}: {mem:.2f} MB")

@profile
def run():
    for index,data_name in enumerate(dataset_list): # 每个数据集
        
        if not os.path.exists('./result'):
            os.makedirs('./result')

        # if os.path.isfile('./result/' + data_name + '.csv'):
        #     continue
        # if dataset_sorted.iloc[index,2] >= 100000:     # 维数
        #     continue
        # if dataset_sorted.iloc[index,3] >= 50:       # 类别数
        #     continue

        print("We are nowing clustering the "+data_name+" dataset.")

        #样本，标签，样本数，维度，聚类数
        X, y_true, N, dim, k = Ifuns.loadmat("{}{}.mat".format(data_path,data_name))
        X = X.astype(np.float64)
        knn = knn_list[data_name]
        with open('./result/'+ data_name + '.csv','w',newline='') as file:
            writer = csv.writer(file,delimiter='\t')
            header = 'acc,nmi,purity,ari,knn,knn time,GLKM time'
            writer.writerow(header.split(','))
            for epoch in range(1):

                print(epoch)

                # NN, NND, max_distance = Gfuns.build_symmetric_knn_graph_union(X, knn) 

                if os.path.isfile(f"./knn_data/{data_name}.npz"):
                    data = np.load(f"./knn_data/{data_name}.npz")
                    NN = data["NN"].astype(np.int32)
                    NND = data["NND"].astype(np.float64)
                    time1 = data['time']
                    print("加载完成")
                else:

                    t1 = time.time()
                    NN, NND = Gfuns.knn_nn(X, min(knn, N-1), squared = True)    # 采用sklearn快速构建近邻图  
                    t2 = time.time()
                    np.savez_compressed(f"./knn_data/{data_name}.npz", NN=NN.astype(np.int32), NND=NND.astype(np.float64), time = t2-t1)
                    time1 = t2-t1
                    print("保存成功")
                # NN = np.array(NN, dtype=np.int32)
                # NND = np.array(NND, dtype=np.float64)  # 如果 NND 是 float64，也可以考虑统一
                
                t2 = time.time()
                print('----------------')
                try:
                    obj = GKM(NN=NN.astype(np.int32), NND=NND.astype(np.float64), c_true = k, debug=0)      ## 这句提示内存爆了，说明是初始化的时候有问题
                    obj.opt()
                    y_pred0 = obj.y_pred
                    if not obj.connected:   ## 如果不连通，就没有写入csv文件，应该改为写入一个特定值
                        print('not connected!')
                        continue
                    y = Ifuns.replace_with_continuous(y_pred0)
                    init_Y = [y]
                    # t3 = time.time()
                    mod = FLK(NN=NN.astype(np.int32), NND=NND.astype(np.float64), max_d=-1, c_true=k, debug=False)
                    mod.opt(ITER=100, init_Y=init_Y)
                    t4 = time.time()
                    y_pred = mod.Y[0]

                    acc_ = Mfuns.accuracy(y_true, y_pred)
                    nmi_ = 	Mfuns.nmi(y_true, y_pred)
                    purity_ = Mfuns.purity(y_true, y_pred)
                    ari_ = Mfuns.ari(y_true, y_pred)
                    writer.writerow([acc_,nmi_,purity_,ari_,knn,time1,t4-t2])
                    print(f'{data_name:<10}:\t{acc_:.3f}\t{nmi_:.3f}\t{time1:.2f}')
                except MemoryError:
                    print("[!] MemoryError: GKM initializaion failed")
                    snapshot_memory("after failure")
                    traceback.print_exc()
                print('begin clustering')


run()