from cython cimport view
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t

# cdef class Householder:
#     cdef INDEX_t m
#     cdef INDEX_t max_n
#     cdef FLOAT_t[::view.contiguous, :] T
#     cdef FLOAT_t[::view.contiguous, :] V
#     cdef INDEX_t k