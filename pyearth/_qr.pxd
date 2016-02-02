from cython cimport view
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t

cdef class UpdatingQT2:
    cdef readonly int m
    cdef readonly int max_n
    cdef readonly Householder2 householder
    cdef readonly int k
    cdef readonly double[::1, :] Q_t
    cpdef void update_qt(UpdatingQT2 self)
    cpdef void update(UpdatingQT2 self, double[:] x)
    cpdef downdate(self)

cdef class Householder2:
    cdef readonly int k
    cdef readonly int m
    cdef readonly int max_n
    cdef readonly double[::1, :] V
    cdef readonly double[::1, :] T
    cdef readonly double[::1] tau
    cdef readonly double[::1, :] work
    cpdef void downdate(Householder2 self)
    cpdef void update_from_column(Householder2 self, double[:] c)
    cpdef void update_v_t(Householder2 self)
    cpdef void left_apply(Householder2 self, double[::1, :] C)
    cpdef void left_apply_transpose(Householder2 self, double[::1, :] C)
    cpdef void right_apply(Householder2 self, double[::1, :] C)
    cpdef void right_apply_transpose(Householder2 self, double[::1, :] C)
    

cdef class Householder:
    cdef readonly INDEX_t m
    cdef readonly INDEX_t max_n
    cdef readonly FLOAT_t[:, :] T
    cdef readonly FLOAT_t[:, :] V
    cdef readonly INDEX_t k