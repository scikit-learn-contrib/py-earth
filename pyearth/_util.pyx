# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np
from libc.math cimport sqrt

cpdef int augmented_normal(cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[FLOAT_t, ndim=2] V, FLOAT_t alpha):
    cdef unsigned int i #@DuplicatedSignature
    cdef unsigned int j #@DuplicatedSignature
    cdef unsigned int k #@DuplicatedSignature
    cdef unsigned int m = y.shape[0]
    cdef unsigned int p = V.shape[0]
    cdef unsigned int n = p - 1
    cdef FLOAT_t tmp #@DuplicatedSignature
    
    #Zero out V
    for i in range(p):
        for j in range(p):
            V[i,j] = 0.0
    
    #Fill in the first n rows and columns (0:n)
    for i in range(m):
        for j in range(n):
            if X[i,j] == 0:
                continue
            for k in range(n):
                V[j,k] += X[i,j] * X[i,k]
                
    #Regularize
    for j in range(n):
        V[j,j] += alpha
                
    #Fill in row and column n
    for i in range(m):
        tmp = y[i]
        for j in range(n):
            V[n,j] += X[i,j] * tmp
            V[j,n] += X[i,j] * tmp
        V[n,n] += tmp * tmp


cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty):
    return mse / ((1 - ((basis_size + penalty*(basis_size - 1))/data_size)) ** 2)


cpdef reorderxby(cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=2] B, cnp.ndarray[FLOAT_t,ndim=2] B_orth, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[INT_t, ndim=1] order, cnp.ndarray[INT_t, ndim=1] inv):
    #TODO: This is a bottleneck for large m.  Optimize row copies and swaps with BLAS.
    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int k
    cdef unsigned int l
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n = X.shape[1]
    cdef unsigned int p = B.shape[1]
    cdef unsigned int p_orth = B_orth.shape[1]
    cdef cnp.ndarray[FLOAT_t, ndim=1] xrow = np.empty(shape=n)
    cdef cnp.ndarray[FLOAT_t, ndim=1] brow = np.empty(shape=p)
    cdef cnp.ndarray[FLOAT_t, ndim=1] borthrow = np.empty(shape=p_orth)
    cdef FLOAT_t tmp
    cdef unsigned int idx
    
    for i in range(m):
        
        for l in range(n):
            xrow[l] = X[i,l] #TODO: dcopy
        for l in range(p):
            brow[l] = B[i,l] #TODO: dcopy
        for l in range(p_orth):
            borthrow[l] = B_orth[i,l] #TODO: dcopy
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
            for l in range(p_orth):
                B_orth[j,l] = B_orth[k,l]
            y[j] = y[k]
            inv[j] = inv[k]
            j = k
        for l in range(n):
            X[j,l] = xrow[l]
        for l in range(p):
            B[j,l] = brow[l]
        for l in range(p_orth):
            B_orth[j,l] = borthrow[l]
        y[j] = tmp
        inv[j] = idx

cpdef FLOAT_t fastr(cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1]y, unsigned int k):
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


cdef update_uv(FLOAT_t last_candidate, FLOAT_t candidate, unsigned int candidate_idx, 
                unsigned int last_candidate_idx, unsigned int last_last_candidate_idx, unsigned int k, unsigned int variable, unsigned int parent,
                cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[FLOAT_t, ndim=2] B,
                FLOAT_t *y_cum, cnp.ndarray[FLOAT_t, ndim=1] B_cum, cnp.ndarray[FLOAT_t, ndim=1] u, 
                cnp.ndarray[FLOAT_t, ndim=1] v, cnp.ndarray[FLOAT_t, ndim=1] delta):
    
    #TODO: BLAS
    #TODO: Optimize
    cdef FLOAT_t diff
    cdef FLOAT_t delta_squared
    cdef FLOAT_t delta_y
    cdef FLOAT_t float_tmp
    
    diff = last_candidate - candidate
    delta_squared = 0.0
    delta_y = 0.0
    for j in range(last_last_candidate_idx+1,last_candidate_idx+1):
        y_cum[0] += y[j]
        for h in range(k+1):#TODO: BLAS
            B_cum[h] += B[j,h]
        B_cum[k+1] += B[j,k+1]*B[j,parent]
    delta_y += diff * y_cum[0]
    delta_squared = (diff**2)*(last_candidate_idx+1)
    for j in range(last_candidate_idx+1,candidate_idx):
        float_tmp = (X[j,variable] - candidate) * B[j, parent]
        delta[j] = float_tmp
        delta_squared += float_tmp**2
        delta_y += float_tmp * y[j]

    #Compute the u vector
    u[0:k+2] = np.dot(delta[last_candidate_idx+1:candidate_idx],B[last_candidate_idx+1:candidate_idx,0:k+2]) #TODO: BLAS
    u[0:k+2] += diff*B_cum
    B_cum[k+1] += B_cum[parent] * diff
    B[last_candidate_idx+1:candidate_idx,k+1] += delta[last_candidate_idx+1:candidate_idx] #TODO: BLAS
    float_tmp = u[k+1] * 2
    float_tmp += delta_squared
    float_tmp = sqrt(float_tmp)
    u[k+1] = float_tmp
    u[k+2] = delta_y / float_tmp
    u[0:k+1] /= float_tmp #TODO: BLAS
    
    #Compute the v vector, which is just u with element k+1 zeroed out
    v[:] = u[:]#TODO: BLAS
    v[k+1] = 0




