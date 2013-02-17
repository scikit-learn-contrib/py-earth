cimport numpy as np
ctypedef np.float64_t FLOAT_t
ctypedef np.int_t NP_INT_t
ctypedef np.uint8_t NP_BOOL_t

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)
