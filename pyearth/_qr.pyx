# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True
import numpy as np
from scipy.linalg.cython_lapack cimport dlarfg, dlarft, dlarfb
from scipy.linalg.cython_blas cimport dcopy

cdef void dlarfg_(double* alpha, double[:] x, double* tau):
    cdef int N = x.shape[0] + 1
    cdef int incx = x.strides[0] // x.itemsize
    dlarfg(&N, alpha, <double *> &(x[0]), &incx, tau)

cpdef dlarfg_p(double alpha, double[:] x):
    cdef double alpha_=alpha, tau
    dlarfg_(&alpha_, x, &tau)
    return tau

def compute_householder(alpha, x):
    s = np.dot(x, x)
    v = x.copy()
    if s == 0:
        tau = 0.
    else:
        t = np.sqrt(alpha ** 2 + s)
        if alpha <= 0:
            v_one = alpha - t
        else:
            v_one = -s / (alpha + t)
        tau = 2 * (v_one ** 2) / (s + (v_one ** 2))
        v *= (1 / v_one)
    return tau, v
# 
def apply_elementary_reflection(tau, v, X):
    p = v.shape[0]
    if len(X.shape) == 1:
        X = X[:, None]
    m, n = X.shape
    vv = np.concatenate([[1], v])[:, None]
    X[(m - p -1):, (n - (m-p)):] -= tau * np.dot( vv, np.dot(vv.T, X[(m - p -1):, (n - (m-p)):]) )

cpdef memoryview_leading_dimension(double[::1,:] X):
    cdef s1, sz
    s1 = X.strides[1]
    sz = X.itemsize
    return s1 // sz


cdef class UpdatingQT2:
    def __init__(UpdatingQT2 self, int m, int max_n, Householder2 householder, 
                 int k, double[::1, :] Q_t):
        self.m = m
        self.max_n = max_n
        self.householder = householder
        self.k = k
        self.Q_t = Q_t
    
    @classmethod
    def alloc(cls, int m, int max_n):
        cdef Householder2 householder = Householder2.alloc(m, max_n)
        cdef double[::1, :] Q_t = np.empty(shape=(max_n, m), dtype=float, order='F')
        return cls(m, max_n, householder, 0, Q_t)
    
    cpdef void update_qt(UpdatingQT2 self):
        # Assume that housholder has already been updated and now Q_t needs to be updated 
        # accordingly
        
        # Zero out the new row of Q_t
        cdef double zero = 0.
        cdef int zero_int = 0
        cdef int N = self.m
        cdef double * y = <double *> &(self.Q_t[self.k, 0])
        cdef int incy = self.max_n
        dcopy(&N, &zero, &zero_int, y, &incy)
        
        # Place a one in the right place
        self.Q_t[self.k, self.k] = 1.
        
        # Apply the householder transformation
        self.householder.right_apply_transpose(self.Q_t[self.k:self.k+1, :])
        
        self.k += 1
    
    cpdef void update(UpdatingQT2 self, double[:] x):
        # Updates householder, then calls 
        # update_qt
        self.householder.update_from_column(x)
        self.update_qt()
    
    cpdef downdate(self):
        self.householder.downdate()
        self.k -= 1    

cdef class Householder2:
    
    def __init__(Householder2 self, int k, int m, int max_n, 
                 double[::1, :] V, double[::1, :] T, double[::1] tau, double[::1, :] work):
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
        cdef double[::1, :] V = np.empty(shape=(m, max_n), dtype=float, order='F')
        cdef double[::1, :] T = np.empty(shape=(max_n, max_n), dtype=float, order='F')
        cdef double[::1] tau = np.empty(shape=max_n, dtype=float, order='F')
        cdef double[::1, :] work = np.empty(shape=(m, max_n), dtype=float, order='F')
        return cls(k, m, max_n, V, T, tau, work)
    
    cpdef void downdate(Householder2 self):
        self.k -= 1
    
    cpdef void update_from_column(Householder2 self, double[:] c):
        # Copies c, applies self, then updates V and T
        
        # Copy c into V
        cdef int N = self.m
        cdef double * x = <double *> &(c[0])
        cdef int incx = c.strides[0] / c.itemsize
        cdef double * y = <double *> &(self.V[0, self.k])
        cdef int incy = 1
        dcopy(&N, x, &incx, y, &incy)
        
        # Apply self to new column in V
        self.left_apply_transpose(self.V[:, self.k:self.k+1])
        
        # Update V and T (increments k)
        self.update_v_t()
        
        
    cpdef void update_v_t(Householder2 self):
        # Assume relevant data has been copied into self.V correctly, as by 
        # update_from_column.  Update V and T appropriately.
        cdef int n = self.m - self.k
        cdef double alpha = self.V[self.k, self.k]
        cdef double* x = <double *> &(self.V[(self.k + 1), self.k])
        cdef int incx = self.V.strides[0] // self.V.itemsize
        cdef double tau
        
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
        cdef double * V = <double *> &(self.V[0,0])
        cdef int ldv = self.m
        cdef double * T = <double *> &(self.T[0,0])
        cdef double * tau_arg = <double *> &(self.tau[0])
        cdef int ldt = self.max_n
        dlarft(&direct, &storev, &n, &k, V, &ldv, tau_arg, T, &ldt)
        
        self.k += 1
        
    cpdef void left_apply(Householder2 self, double[::1, :] C):
        cdef char side = 'L'
        cdef char trans = 'N'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef double * V = <double *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef double * T = <double *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef double * C_arg = <double *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef double * work = <double *> &(self.work[0,0])
        cdef int ldwork = self.m
        print C.shape
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
        
    cpdef void left_apply_transpose(Householder2 self, double[::1, :] C):
        cdef char side = 'L'
        cdef char trans = 'T'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef double * V = <double *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef double * T = <double *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef double * C_arg = <double *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef double * work = <double *> &(self.work[0,0])
        cdef int ldwork = self.m
        
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
    
    cpdef void right_apply(Householder2 self, double[::1, :] C):
        cdef char side = 'R'
        cdef char trans = 'N'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef double * V = <double *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef double * T = <double *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef double * C_arg = <double *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef double * work = <double *> &(self.work[0,0])
        cdef int ldwork = self.m
        
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
        
    cpdef void right_apply_transpose(Householder2 self, double[::1, :] C):
        cdef char side = 'R'
        cdef char trans = 'T'
        cdef char direct = 'F'
        cdef char storev = 'C'
        cdef int M = C.shape[0]
        cdef int N = C.shape[1]
        cdef int K = self.k
        cdef double * V = <double *> &(self.V[0, 0])
        cdef int ldv = self.m
        cdef double * T = <double *> &(self.T[0, 0])
        cdef int ldt = self.max_n
        cdef double * C_arg = <double *> &(C[0, 0])
        cdef int ldc = C.strides[1] // C.itemsize
        cdef double * work = <double *> &(self.work[0,0])
        cdef int ldwork = self.m
        
        dlarfb(&side, &trans, &direct, &storev, &M, &N, &K, 
               V, &ldv, T, &ldt, C_arg, &ldc, work, &ldwork)
        
        
def dot0(x, y):
    return np.dot(x, y)
def dot1(x, y):
    return np.dot(x, y)
def dot2(x, y):
    return np.dot(x, y)
def dot3(x, y):
    return np.dot(x, y)
def dot4(x, y):
    return np.dot(x, y)
def dot5(x, y):
    return np.dot(x, y)
def dot6(x, y):
    return np.dot(x, y)
def dot7(x, y):
    return np.dot(x, y)
def dot8(x, y):
    return np.dot(x, y)
def dot9(x, y):
    return np.dot(x, y)
def dot10(x, y):
    return np.dot(x, y)
def dot11(x, y):
    return np.dot(x, y)

cdef class Householder:
    def __init__(self, m, max_n):
        self.m = m
        self.max_n = max_n
        self.T = np.zeros(shape=(max_n, max_n), dtype=float)
        self.V = np.zeros(shape=(self.m, self.max_n))
        self.k = 0
    
    def push_from_column(self, alpha, x):
        s = dot0(x, x)
        np.asarray(self.V)[(self.k + 1):, self.k] = x
        self.V[self.k, self.k] = 1.
        if s == 0:
            tau = 0.
        else:
            t = np.sqrt(alpha ** 2 + s)
            if alpha <= 0:
                v_one = alpha - t
            else:
                v_one = -s / (alpha + t)
            tau = 2 * (v_one ** 2) / (s + (v_one ** 2))
            np.asarray(self.V)[self.k + 1:, self.k] *= (1 / v_one)
        if self.k > 0:
            T_ = self.T[:self.k,:self.k]
            np.asarray(self.T)[:self.k,self.k] = -tau * dot1(dot2(T_, self.V[:,:self.k].T), self.V[:,self.k])
            self.T[self.k, self.k] = tau
        else:
            self.T[0,0] = tau
        self.k += 1
        
    def push_elementary_reflection(self, tau, v):
        np.asarray(self.V)[self.k + 1:,self.k] = v
        self.V[self.k, self.k] = 1.
        if self.k > 0:
            T_ = self.T[:self.k,:self.k]
            np.asarray(self.T)[:self.k,self.k] = -tau * np.dot(np.dot(T_, self.V[:,:self.k].T), self.V[:,self.k])
            self.T[self.k, self.k] = tau
        else:
            self.T[0,0] = tau
        self.k += 1
    
    def pop_elementary_reflection(self):
        self.k -= 1
    
    def reset(self):
        self.k = 0
        
    def apply(self, X):
        return X - dot3(self.V[:,:self.k], dot4(self.T[:self.k,:self.k], dot5(self.V[:,:self.k].T, X)))
    
    def apply_to_transpose(self, X):
        return X - (dot6(self.V[:,:self.k], dot7(self.T[:self.k,:self.k], dot8(self.V[:,:self.k].T, X.T)))).T
    
    def apply_transpose(self, X):
        return X - dot9(self.V[:,:self.k], dot10(self.T[:self.k,:self.k].T, dot11(self.V[:,:self.k].T, X)))
    
    def apply_transpose_to_transpose(self, X):
        return X - (np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k].T, np.dot(self.V[:,:self.k].T, X.T)))).T
    

class UpdatingQT(object):
    def __init__(self, m, max_n):
        self.m = m
        self.max_n = max_n
        self.householder = Householder(self.m, self.max_n)
        self.Q_t = np.zeros(shape=(self.max_n, self.m), dtype=float)
        self.k = 0
    
    def update(self, x):
        x_ = self.householder.apply_transpose(x)
        
#         tau, v = compute_householder(x_[self.k], x_[(self.k + 1):])
        
        self.householder.push_from_column(x_[self.k], x_[(self.k + 1):])
        self.Q_t[self.k, self.k] = 1.
        np.asarray(self.Q_t)[self.k, :] = self.householder.apply(self.Q_t[self.k, :])
        self.k += 1
    
    
    def downdate(self):
        self.householder.pop_elementary_reflection()
        self.Q_t[self.k - 1, :] = 0.
        self.k -= 1

if __name__ == '__main__':
    m = 10
    n = 3
    X = np.random.normal(size=(m, n))
    u = UpdatingQT(m, n)
    Q, R = np.linalg.qr(X, mode='reduced')[:2]
    u.update(X[:,0])
    u.update(X[:,1])
    u.update(X[:,2])
    
    assert np.max(np.abs(np.abs(u.Q_t) - np.abs(Q.T))) < .0000000000001
    
    X2 = X.copy()
    X2[:,2] = np.random.normal(size=m)
    u.downdate()
    u.update(X2[:,2])
    Q2 = np.linalg.qr(X2, mode='reduced')[0]
    assert np.max(np.abs(np.abs(u.Q_t) - np.abs(Q2.T))) < .0000000000001
    
    print 'Success!'
    
