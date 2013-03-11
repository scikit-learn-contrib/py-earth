cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint8_t BOOL_t

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)

cpdef inline FLOAT_t gcv_adjust(unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)

cpdef str_pad(string, length)

cpdef ascii_table(header, data)