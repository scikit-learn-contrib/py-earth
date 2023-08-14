cimport numpy as cnp
from ._types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t

cdef FLOAT_t log2(FLOAT_t x)

cpdef apply_weights_2d(cnp.ndarray[FLOAT_t, ndim=2] B,
                       cnp.ndarray[FLOAT_t, ndim=1] weights)

cpdef apply_weights_slice(cnp.ndarray[FLOAT_t, ndim=2] B,
                          cnp.ndarray[FLOAT_t, ndim=1] weights, INDEX_t column)

cpdef apply_weights_1d(cnp.ndarray[FLOAT_t, ndim=1] y,
                       cnp.ndarray[FLOAT_t, ndim=1] weights)

cpdef FLOAT_t gcv(FLOAT_t mse,
                  FLOAT_t basis_size, FLOAT_t data_size,
                  FLOAT_t penalty)

cpdef FLOAT_t gcv_adjust(FLOAT_t basis_size, FLOAT_t data_size, FLOAT_t penalty)

cpdef str_pad(string, length)

cpdef ascii_table(header, data, print_header=?, print_footer=?)
