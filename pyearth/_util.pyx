# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np
from libc.math cimport sqrt

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty):
    return mse / gcv_adjust(basis_size, data_size, penalty)

cpdef inline FLOAT_t gcv_adjust(unsigned int basis_size, unsigned int data_size, FLOAT_t penalty):
    return ((1 - ((basis_size + penalty*(basis_size - 1))/data_size)) ** 2)

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
        
cpdef str_pad(string, length):
    if len(string) >= length:
        return string[0:length]
    pad = length - len(string)
    return string + ' '*pad

cpdef ascii_table(header, data):
    '''
    header - list of strings representing the header row
    data - list of lists of strings representing data rows
    '''
    m = len(data)
    n = len(header)
    column_widths = [len(head) for head in header]
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if len(col) > column_widths[j]:
                column_widths[j] = len(col)
    
    for j in range(n):
        column_widths[j] += 1
    
    result = ''
    for j, col_width in enumerate(column_widths):
        result += '-'*col_width + '-'
    result += '\n'
    for j, head in enumerate(header):
        result += str_pad(head,column_widths[j]) + ' '
    result += '\n'
    for j, col_width in enumerate(column_widths):
        result += '-'*col_width + '-'
    for i, row in enumerate(data):
        result += '\n'
        for j, item in enumerate(row):
            result += str_pad(item,column_widths[j]) + ' '
    result += '\n'
    for j, col_width in enumerate(column_widths):
        result += '-'*col_width + '-'
    return result
    
    
    
    
    
    
