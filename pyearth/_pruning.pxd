cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint8_t BOOL_t
from _basis cimport Basis
from _record cimport PruningPassRecord

cdef class PruningPasser:
    cdef cnp.ndarray X
    cdef cnp.ndarray B
    cdef cnp.ndarray y
    cdef unsigned int m
    cdef unsigned int n
    cdef Basis basis
    cdef FLOAT_t penalty
    cdef FLOAT_t sst
    cdef PruningPassRecord record
    
    cpdef run(PruningPasser self)
    
    cpdef PruningPassRecord trace(PruningPasser self)