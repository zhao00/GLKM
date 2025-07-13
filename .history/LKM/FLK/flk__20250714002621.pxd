from libcpp.vector cimport vector
from libcpp cimport bool

# cdef extern from "CppFuns.cpp":
#     pass

cdef extern from "flk.cpp":
    pass

cdef extern from "flk.h":
    cdef cppclass flk:

        vector[vector[int]] Y
        vector[unsigned int] n_iter
        
        flk() except +
        flk(vector[vector[int]] &NN, vector[vector[double]] &NND, double max_d, int c_true, bool debug) except +
        void opt(int ITER, vector[vector[int]] &init_Y)

        # void __dealloc__()
        # ~flk() 
        void __dealloc__()