cimport numpy as np
ctypedef np.float64_t FLOAT_t
ctypedef np.int_t INT_t
ctypedef np.uint8_t BOOL_t

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)

cpdef reorderxby(np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=2] B, np.ndarray[FLOAT_t, ndim=1] y, np.ndarray[INT_t, ndim=1] order, np.ndarray[INT_t, ndim=1] inv)

cpdef FLOAT_t fastr(np.ndarray[FLOAT_t, ndim=2] X, np.ndarray[FLOAT_t, ndim=1]y, unsigned int k)