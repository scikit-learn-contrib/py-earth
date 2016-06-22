cimport numpy as cnp
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t
from _basis cimport Basis
from _record cimport PruningPassRecord

cdef class PruningPasser:
    cdef cnp.ndarray X
    cdef cnp.ndarray missing
    cdef cnp.ndarray B
    cdef cnp.ndarray y
    cdef cnp.ndarray sample_weight
    cdef int verbose
    cdef cnp.ndarray output_weight
    cdef public dict feature_importance
    
    cdef INDEX_t m
    cdef INDEX_t n
    cdef Basis basis
    cdef FLOAT_t penalty
    cdef FLOAT_t sst
    cdef PruningPassRecord record

    cpdef run(PruningPasser self)

    cpdef PruningPassRecord trace(PruningPasser self)
