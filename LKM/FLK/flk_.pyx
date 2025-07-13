cimport numpy as np
import numpy as np
np.import_array()

#from FeiPub import Ifuns

from .flk_ cimport flk

cdef class PyFLK:
    cdef flk *c_flk
    cdef int N
    cdef int c_true

    def __cinit__(self, int c_true, bool debug, np.ndarray[int, ndim=2] NN, np.ndarray[double, ndim=2] NND, double max_d):

        self.c_flk = new flk(NN, NND, max_d, c_true, debug)
        self.N = NN.shape[0]

        self.c_true = c_true

    def opt(self, int ITER, init_Y):
        self.c_flk.opt(ITER, init_Y)

    @property
    def n_iter(self):
        return np.array(self.c_flk.n_iter)

    @property
    def Y(self):
        return np.array(self.c_flk.Y)

    # def __dealloc__(self):
    #     if self.c_flk is not None:
    #         del self.c_flk
    #         self.c_flk = None

    def __dealloc__(self):

        del self.c_flk