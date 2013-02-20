# distutils: language = c
#cython: boundscheck=True
#cython: wraparound=True
import numpy as np

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty):
    return mse / ((1 - ((basis_size + penalty*(basis_size - 1))/data_size)) ** 2)


cpdef reorderxby(np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=2] B, np.ndarray[FLOAT_t, ndim=1] y, np.ndarray[INT_t, ndim=1] order, np.ndarray[INT_t, ndim=1] inv):
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int p = B.shape[1]
    cdef np.ndarray[FLOAT_t, ndim=1] xrow = np.empty(shape=n)
    cdef np.ndarray[FLOAT_t, ndim=1] brow = np.empty(shape=p)
    cdef FLOAT_t tmp
    cdef unsigned int idx
    
    for i in range(m):
        
        for l in range(n):
            xrow[l] = X[i,l]
        for l in range(p):
            brow[l] = B[i,l]
        tmp = y[i]
        idx = inv[i]
        j = i
        while True:
            k = order[j]
            order[j] = j
            if k == i:
                break
            for l in range(n):
                X[j,l] = X[k,l]
            for l in range(p):
                B[j,l] = B[k,l]
            y[j] = y[k]
            inv[j] = inv[k]
            j = k
        for l in range(n):
            X[j,l] = xrow[l]
        for l in range(p):
            B[j,l] = brow[l]
        y[j] = tmp
        inv[j] = idx

cpdef FLOAT_t fastr(np.ndarray[FLOAT_t, ndim=2] X, np.ndarray[FLOAT_t, ndim=1]y, unsigned int k):
    cdef unsigned int i #@DuplicatedSignature
    cdef unsigned int m = X.shape[0] #@DuplicatedSignature
    for i in range(m):
        tmp = X[i,k]
        X[i,k] = y[i]
        y[i] = tmp
        
    R = np.linalg.qr(X[:,0:(k+1)],mode='r')

    for i in range(m):
        tmp = X[i,k]
        X[i,k] = y[i]
        y[i] = tmp
        
    return R[k,k] ** 2
