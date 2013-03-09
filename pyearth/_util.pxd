cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint8_t BOOL_t

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)

cpdef inline FLOAT_t gcv_adjust(unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)

cpdef reorderxby(cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=2] B, cnp.ndarray[FLOAT_t,ndim=2] B_orth, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[INT_t, ndim=1] order, cnp.ndarray[INT_t, ndim=1] inv)

cpdef str_pad(string, length)

cpdef ascii_table(header, data)