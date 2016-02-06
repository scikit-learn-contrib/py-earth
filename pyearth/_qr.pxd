from cython cimport view
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t

cdef class UpdatingQT:
    cdef readonly int m
    cdef readonly int max_n
    cdef readonly Householder householder
    cdef readonly int k
    cdef readonly FLOAT_t[::1, :] Q_t
    cdef readonly FLOAT_t zero_tol
    cdef readonly BOOL_t[::1] dependent_cols
    cpdef void update_qt(UpdatingQT self, bint dependent)
    cpdef void update(UpdatingQT self, FLOAT_t[:] x)
    cpdef void downdate(UpdatingQT self)
    cpdef void reset(UpdatingQT self)

cdef class Householder:
    cdef readonly int k
    cdef readonly int m
    cdef readonly int max_n
    cdef readonly FLOAT_t[::1, :] V
    cdef readonly FLOAT_t[::1, :] T
    cdef readonly FLOAT_t[::1] tau
    cdef readonly FLOAT_t[::1] beta
    cdef readonly FLOAT_t[::1, :] work
    cdef readonly FLOAT_t zero_tol
    cpdef void downdate(Householder self)
    cpdef void reset(Householder self)
    cpdef bint update_from_column(Householder self, FLOAT_t[:] c)
    cpdef bint update_v_t(Householder self)
    cpdef void left_apply(Householder self, FLOAT_t[::1, :] C)
    cpdef void left_apply_transpose(Householder self, FLOAT_t[::1, :] C)
    cpdef void right_apply(Householder self, FLOAT_t[::1, :] C)
    cpdef void right_apply_transpose(Householder self, FLOAT_t[::1, :] C)
