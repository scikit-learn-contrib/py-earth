#cython: cdivision=True
#cython: boundscheck=True
#cython: wraparound=True

from libc.math cimport sqrt
from libc.math cimport abs as cabs
import numpy as np

cdef inline FLOAT_t hypot(FLOAT_t x,FLOAT_t y):
    cdef FLOAT_t t
    x = cabs(x)
    y = cabs(y)
    t = x if x < y else y
    x = x if x > y else y
    t = t/x
    return x*sqrt(1+t*t)

cdef inline FLOAT_t rypot(FLOAT_t x,FLOAT_t y):
    cdef FLOAT_t t
    x = cabs(x)
    y = cabs(y)
    t = x if x < y else y
    x = x if x > y else y
    t = t/x
    return x*sqrt(1-t*t)

cpdef cholupdate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x):
    '''
    Update the upper triangular Cholesky factor R with the rank 1 addition
    implied by x such that:
    R_'R_ = R'R + outer(x,x)
    where R_ is the upper triangular Cholesky factor R after updating.  Note
    that both x and R are modified in place.
    '''
    cdef unsigned int p
    cdef unsigned int k
    cdef unsigned int i
    cdef FLOAT_t r
    cdef FLOAT_t c
    cdef FLOAT_t s
    cdef FLOAT_t a
    cdef FLOAT_t b

    p = <unsigned int>len(x)
    for k in range(p):
        r = hypot(R[<unsigned int>k,<unsigned int>k], x[<unsigned int>k])
        c = r / R[<unsigned int>k,<unsigned int>k]
        s = x[<unsigned int>k] / R[<unsigned int>k,<unsigned int>k]
        R[<unsigned int>k,<unsigned int>k] = r
        #TODO: Use BLAS instead of inner for loop
        for i in range(<unsigned int>(k+1),<unsigned int>p):
            R[<unsigned int>k,<unsigned int>i] = (R[<unsigned int>k,<unsigned int>i] + s*x[<unsigned int>i]) / c
            x[<unsigned int>i] = c * x[<unsigned int>i] - s * R[<unsigned int>k,<unsigned int>i]

cpdef choldowndate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x):
    '''
    Update the upper triangular Cholesky factor R with the rank 1 subtraction
    implied by x such that:
    R_'R_ = R'R - outer(x,x)
    where R_ is the upper triangular Cholesky factor R after updating.  Note
    that both x and R are modified in place.
    '''
    cdef unsigned int p
    cdef unsigned int k
    cdef unsigned int i
    cdef FLOAT_t r
    cdef FLOAT_t c
    cdef FLOAT_t s

    p = <unsigned int>len(x)
    for k in range(p):
        r = rypot(R[<unsigned int>k,<unsigned int>k], x[<unsigned int>k])
        c = r / R[<unsigned int>k,<unsigned int>k]
        s = x[<unsigned int>k] / R[<unsigned int>k,<unsigned int>k]
        R[<unsigned int>k,<unsigned int>k] = r
        #TODO: Use BLAS instead of inner for loop
        for i in range(<unsigned int>(k+1),<unsigned int>p):
            R[<unsigned int>k,<unsigned int>i] = (R[<unsigned int>k,<unsigned int>i] - s*x[<unsigned int>i]) / c
            x[<unsigned int>i] = c * x[<unsigned int>i] - s * R[<unsigned int>k,<unsigned int>i]
