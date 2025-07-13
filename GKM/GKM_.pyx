# distutils: language = c++
from GKM_ cimport GKM
import numpy as np

# Define a Python class that wraps around the C++ GKM class
cdef class PyGKM:
    cdef GKM* cpp_gkm

    def __cinit__(self, NN,  NND,  c_true, debug=0):

        self.cpp_gkm = new GKM(NN, NND, c_true, debug=0)

    def opt(self):
        self.cpp_gkm.opt()

    @property
    def y_pred(self):
        return np.array(self.cpp_gkm.labels)

    @property
    def loss(self):
        return np.array(self.cpp_gkm.loss) 

    @property
    def c_now(self):
        return self.cpp_gkm.c_now

    @property
    def connected(self):
        return self.cpp_gkm.connected

    # def __dealloc__(self):

    #     del self.cpp_gkm
    #         #self.cpp_gkm = None
