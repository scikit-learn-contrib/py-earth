# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np
from libc.math cimport sqrt, log

cdef FLOAT_t log2(FLOAT_t x):
    return log(x) / log(2.0)

cpdef FLOAT_t gcv(FLOAT_t mse, INDEX_t basis_size, INDEX_t data_size, FLOAT_t penalty):
    return mse * gcv_adjust(basis_size, data_size, penalty)

cpdef FLOAT_t gcv_adjust(INDEX_t basis_size, INDEX_t data_size, FLOAT_t penalty):
    return 1.0 / ((1 - ((basis_size + penalty*(basis_size - 1))/data_size)) ** 2)

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
    
    
    
    
    
    
