
cimport numpy as np
ctypedef np.float64_t FLOAT_t

cpdef cholupdate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x)

cpdef choldowndate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x)