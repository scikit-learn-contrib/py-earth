cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint8_t BOOL_t

cpdef int augmented_normal(cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[FLOAT_t, ndim=2] V, FLOAT_t alpha)

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty)

cpdef reorderxby(cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=2] B, cnp.ndarray[FLOAT_t,ndim=2] B_orth, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[INT_t, ndim=1] order, cnp.ndarray[INT_t, ndim=1] inv)

cpdef FLOAT_t fastr(cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1]y, unsigned int k)

cdef update_uv(FLOAT_t last_candidate, FLOAT_t candidate, unsigned int candidate_idx, 
                unsigned int last_candidate_idx, unsigned int last_last_candidate_idx, unsigned int k, unsigned int variable, unsigned int parent,
                cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] y, cnp.ndarray[FLOAT_t, ndim=2] B,
                FLOAT_t *y_cum, cnp.ndarray[FLOAT_t, ndim=1] B_cum, cnp.ndarray[FLOAT_t, ndim=1] u, 
                cnp.ndarray[FLOAT_t, ndim=1] v, cnp.ndarray[FLOAT_t, ndim=1] delta)