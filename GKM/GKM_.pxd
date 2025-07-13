from libcpp.vector cimport vector
from libcpp cimport bool
#from libcpp.set cimport set as cpp_set
#from libcpp.tuple cimport tuple as cpp_tuple

cdef extern from "GKM.cpp":
    pass

cdef extern from "GKM.h":
    cdef cppclass GKM:
        GKM() except +
        GKM(vector[vector[int]] &NN, vector[vector[double]] &NND, int c_true, int debug) except +
        void opt()
        #void __dealloc__()

        vector[int] labels
        #cpp_set[cpp_tuple[double, int, int]] tree
        int c_now
        double loss
        bool connected