# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True
import numpy as np
from scipy.linalg.cython_lapack cimport dlarfg, dlarft, dlarfb
from scipy.linalg.cython_blas cimport dcopy

cdef class UpdatingQT:
    def __init__(UpdatingQT self, int m, int max_n, Householder householder, 
                 int k, FLOAT_t[::1, :] Q_t):
        self.m = m
        self.max_n = max_n
        self.householder = householder
        self.k = k
        self.Q_t = Q_t
    
    @classmethod
    def alloc(cls, int m, int max_n):
        cdef Householder householder = Householder.alloc(m, max_n)
        cdef FLOAT_t[::1, :] Q_t = np.empty(shape=(max_n, m), dtype=float, order='F')
        return cls(m, max_n, householder, 0, Q_t)
    
    cpdef void update_qt(UpdatingQT self):
        # Assume that housholder has already been updated and now Q_t needs to be updated 
        # accordingly
        
        # Zero out the new row of Q_t
        cdef FLOAT_t zero = 0.
        cdef int zero_int = 0
        cdef int N = self.m
        cdef FLOAT_t * y = <FLOAT_t *> &(self.Q_t[self.k, 0])
        cdef int incy = self.max_n
        dcopy(&N, &zero, &zero_int, y, &incy)
        
        # Place a one in the right place
        self.Q_t[self.k, self.k] = 1.
        
        # Apply the householder transformation
        self.householder.right_apply_transpose(self.Q_t[self.k:self.k+1, :])
        
        self.k += 1
    
    cpdef void update(UpdatingQT self, FLOAT_t[:] x):
        # Updates householder, then calls 
        # update_qt
        self.householder.update_from_column(x)
        self.update_qt()
    
    cpdef void downdate(UpdatingQT self):
        self.householder.downdate()
        self.k -= 1
        
    cpdef void reset(UpdatingQT self):
        self.householder.reset()
        self.k = 0
    
cdef class Householder:
    
    def __init__(Householder self, int k, int m, int max_n, 
                 FLOAT_t[::1, :] V, FLOAT_t[::1, :] T, FLOAT_t[::1] tau, FLOAT_t[::1, :] work):
        self.k = k
        self.m = m
        self.max_n = max_n
        self.V = V
        self.T = T
        self.tau = tau
        self.work = work
        
    @classmethod
    def alloc(cls, int m, int max_n):
        cdef int k = 0
        cdef FLOAT_t[::1, :] V = np.empty(shape=(m, max_n), dtype=float, order='F')
        cdef FLOAT_t[::1, :] T = np.empty(shape=(max_n, max_n), dtype=float, order='F')
        cdef FLOAT_t[::1] tau = np.empty(shape=max_n, dtype=float, order='F')
        cdef FLOAT_t[::1, :] work = np.empty(shape=(m, max_n), dtype=float, order='F')
        return cls(k, m, max_n, V, T, tau, work)
    
    cpdef void downdate(Householder self):
        self.k -= 1
    
    cpdef void reset(Householder self):
        self.k = 0
    
    cpdef void update_from_column(Householder self, FLOAT_t[:] c):
        # Copies c, applies self, then updates V and T
        
        # Copy c into V
        cdef int N = self.m
        cdef FLOAT_t * x = <FLOAT_t *> &(c[0])
        cdef int incx = c.strides[0] / c.itemsize
        cdef FLOAT_t * y = <FLOAT_t *> &(self.V[0, self.k])
        cdef int incy = 1
        dcopy(&N, x, &incx, y, &incy)
        
        # Apply self to new column in V
        self.left_apply_transpose(self.V[:, self.k:self.k+1])
        
        # Update V and T (increments k)
        self.update_v_t()
        
        
    cpdef void update_v_t(Householder self):
        # Assume relevant data has been copied into self.V correctly, as by 
        # update_from_column.  Update V and T appropriately.
        cdef int n = self.m - self.k
        cdef FLOAT_t alpha = self.V[self.k, self.k]
        cdef FLOAT_t* x = <FLOAT_t *> &(self.V[(self.k + 1), self.k])
        cdef int incx = self.V.strides[0] // self.V.itemsize
        cdef FLOAT_t tau
        
        # Compute the householder reflection
        dlarfg(&n, &alpha, x, &incx, &tau)
        
        # Add the new householder reflection to the 
        # block reflector
        # TODO: Currently requires recalculating all of T
        # Should be updated to use BLAS instead to calculate 
        # just the new column of T
        self.V[self.k, self.k] = 1.
        self.V[:self.k, self.k] = 0.
        self.tau[self.k] = tau
        cdef char direct = 'F'
        cdef char storev = 'C'
        n = self.m
        cdef int k = self.k + 1
        cdef FLOAT_t * V = <FLOAT_t *> &(self.V[0,0])
        cdef int ldv = self.m
        cdef FLOAT_t * T = <FLOAT_t *> &(self.T[0,0])
        cdef FLOAT_t * tau_arg = <FLOAT_t *> &(self.tau[0])
        cdef int ldt = self.max_n
        dlarft(&direct, &storev, &n, &k, V, &ldv, tau_arg, T, &ldt)
        
        self.k += 1
        
    cpdef void left_apply(Householder self, FLOAT_t[::1, :] C):
        cdef char side = 'L'
        cdef char trans = 'N'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef FLOAT_t * V = <FLOAT_t *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef FLOAT_t * T = <FLOAT_t *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef FLOAT_t * C_arg = <FLOAT_t *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef FLOAT_t * work = <FLOAT_t *> &(self.work[0,0])
        cdef int ldwork = self.m
        print C.shape
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
        
    cpdef void left_apply_transpose(Householder self, FLOAT_t[::1, :] C):
        cdef char side = 'L'
        cdef char trans = 'T'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef FLOAT_t * V = <FLOAT_t *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef FLOAT_t * T = <FLOAT_t *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef FLOAT_t * C_arg = <FLOAT_t *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef FLOAT_t * work = <FLOAT_t *> &(self.work[0,0])
        cdef int ldwork = self.m
        
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
    
    cpdef void right_apply(Householder self, FLOAT_t[::1, :] C):
        cdef char side = 'R'
        cdef char trans = 'N'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef FLOAT_t * V = <FLOAT_t *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef FLOAT_t * T = <FLOAT_t *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef FLOAT_t * C_arg = <FLOAT_t *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef FLOAT_t * work = <FLOAT_t *> &(self.work[0,0])
        cdef int ldwork = self.m
        
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
        
    cpdef void right_apply_transpose(Householder self, FLOAT_t[::1, :] C):
        cdef char side = 'R'
        cdef char trans = 'T'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef FLOAT_t * V = <FLOAT_t *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef FLOAT_t * T = <FLOAT_t *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef FLOAT_t * C_arg = <FLOAT_t *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef FLOAT_t * work = <FLOAT_t *> &(self.work[0,0])
        cdef int ldwork = self.m
        
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
#         
