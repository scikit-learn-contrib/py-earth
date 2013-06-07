cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INT_t
ctypedef cnp.ulong_t INDEX_t
ctypedef cnp.uint8_t BOOL_t

cdef FLOAT_t log2(FLOAT_t x)

cpdef FLOAT_t gcv(FLOAT_t mse, INDEX_t basis_size, INDEX_t data_size, FLOAT_t penalty)

cpdef FLOAT_t gcv_adjust(INDEX_t basis_size, INDEX_t data_size, FLOAT_t penalty)

cpdef str_pad(string, length)

cpdef ascii_table(header, data)