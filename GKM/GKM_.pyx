# distutils: language = c++
cimport numpy as np
import numpy as np
np.import_array()
from GKM_ cimport GKM

# Define a Python class that wraps around the C++ GKM class
cdef class PyGKM:
    cdef GKM *cpp_gkm

    def __cinit__(self, np.ndarray[int, ndim=2] NN, np.ndarray[double, ndim=2]  NND, int c_true, int debug):

        self.cpp_gkm = new GKM(NN, NND, c_true, debug)

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
