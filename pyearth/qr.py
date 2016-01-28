'''
Created on Jan 26, 2016

@author: jason
'''
import numpy as np

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

def apply_elementary_reflection(tau, v, X):
    p = v.shape[0]
    if len(X.shape) == 1:
        X = X[:, None]
    m, n = X.shape
    vv = np.concatenate([[1], v])[:, None]
    X[(m - p -1):, (n - (m-p)):] -= tau * np.dot( vv, np.dot(vv.T, X[(m - p -1):, (n - (m-p)):]) )


class Householder(object):
    def __init__(self, m, max_n):
        self.m = m
        self.max_n = max_n
        self.T = np.zeros(shape=(max_n, max_n), dtype=float)
        self.V = np.zeros(shape=(self.m, self.max_n))
        self.k = 0
        
    def push_elementary_reflection(self, tau, v):
        self.V[self.k + 1:,self.k] = v
        self.V[self.k, self.k] = 1.
        if self.k > 0:
            T_ = self.T[:self.k,:self.k]
            self.T[:self.k,self.k] = -tau * np.dot(np.dot(T_, self.V[:,:self.k].T), self.V[:,self.k])
            self.T[self.k, self.k] = tau
        else:
            self.T[0,0] = tau
        self.k += 1
    
    def pop_elementary_reflection(self):
        self.k -= 1
    
    def apply(self, X, k=None):
        return X - np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k], np.dot(self.V[:,:self.k].T, X)))
    
    def apply_to_transpose(self, X, k=None):
        return X - (np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k], np.dot(self.V[:,:self.k].T, X.T)))).T
    
    def apply_transpose(self, X, k=None):
        return X - np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k].T, np.dot(self.V[:,:self.k].T, X)))
    
    def apply_transpose_to_transpose(self, X, k=None):
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
        tau, v = compute_householder(x_[self.k], x_[(self.k + 1):])
        
        self.householder.push_elementary_reflection(tau, v)
        self.Q_t[self.k, self.k] = 1.
        self.Q_t[self.k, :] = self.householder.apply(self.Q_t[self.k, :])
#         self.Q_t[self.k, :]
#         apply_elementary_reflection(tau, v, self.Q_t[self.k, :])
        
        
        self.k += 1
    
    
    def downdate(self):
        self.householder.pop_elementary_reflection()
        self.Q_t[self.k - 1, :] = 0.
        self.k -= 1
        
# class UpdatingQT(object):
#     def __init__(self, m, max_n):
#         self.m = m
#         self.max_n = max_n
#         self.T = np.zeros(shape=(self.max_n, self.max_n), dtype=float)
#         self.V = np.zeros(shape=(self.m, self.max_n))
#         self.Q_t = np.zeros(shape=(self.max_n, self.m), dtype=float)
#         self.k = 0
# 
#     def update(self, x):
#         # Apply existing transformation to x
#         x -= np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k], np.dot(self.V[:,:self.k].T, x)))
#         
#         # Compute the new reflection
#         tau, v = compute_householder(x[self.k], x[(self.k + 1):])
#         
#         # Add a new row to Q_t and apply existing transformation
#         self.Q_t[self.k, self.k] = 1.
#         self.Q_t[self.k, :] -= (np.dot(self.V[:,:self.k], 
#                                        np.dot(self.T[:self.k,:self.k].T, 
#                                               np.dot(self.V[:,:self.k].T, 
#                                                      self.Q_t[self.k, :].T)))).T
#         
#         # Apply the new householder reflection to the new row of Q_t
#         apply_elementary_reflection(tau, v, self.Q_t[self.k, :])
#         
#         # Update the block householder matrices with the new reflection
#         self.V[self.k + 1:,self.k] = v
#         self.V[self.k, self.k] = 1.
#         if self.k > 0:
#             T_ = self.T[:self.k,:self.k]
#             self.T[:self.k,self.k] = -tau * np.dot(np.dot(T_, self.V[:,:self.k].T), self.V[:,self.k])
#             self.T[self.k, self.k] = tau
#         else:
#             self.T[0,0] = tau
#         
#         self.k += 1
#         
#     def downdate(self):
#         # Can't downdate if there's nothing left to downdate
#         if self.k == 0:
#             raise ValueError
#         
#         # Invert the last elementary reflection on Q_t
#         tau = self.T[self.k - 1, self.k - 1]
#         v = self.V[self.k:, self.k - 1]
#         apply_elementary_reflection(tau, v, self.Q_t.T)
#         
#         # Zero out the appropriate parts of V and T
#         self.V[self.k - 1:, self.k - 1] = 0.
#         self.T[:(self.k - 1), self.k - 2] = 0.
#         self.T[self.k - 1, self.k - 1] = 0.
#         
#         # Decrement k
#         self.k -= 1
# #     def apply_transpose(self, X):
# #         return X - np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k].T, np.dot(self.V[:,:self.k].T, X)))
# #     






if __name__ == '__main__':
    m = 10
    n = 3
    X = np.random.normal(size=(m, n))
    u = UpdatingQT(m, n)
    Q, R = np.linalg.qr(X, mode='reduced')[:2]
    u.update(X[:,0].copy())
    u.update(X[:,1].copy())
    u.update(X[:,2].copy())
    
    assert np.max(np.abs(np.abs(u.Q_t) - np.abs(Q.T))) < .000000001
    
    X2 = X.copy()
    X2[:,2] = np.random.normal(size=m)
    u.downdate()
    u.update(X2[:,2])
    Q2 = np.linalg.qr(X2, mode='reduced')[0]
    assert np.max(np.abs(np.abs(u.Q_t) - np.abs(Q2.T))) < .000000001
    
    print 'Success!'
    
#     print u.Q_t.T
#     print Q

#     print np.dot(u.Q_t, X)
#     print np.dot(Q.T, X)
# #    
#     X2 = X.copy()
#     X2[:, 1] = np.random.normal(size=m)
#     H = Householder(m, n)
#     tau1, v1 = compute_householder(X[0,0], X[1:,0])
#     H.add_elementary_reflection(tau1, v1)
# #     R = X.copy()
#     R = H.apply(X)
#     tau2, v2 = compute_householder(R[1,1], R[2:,1])
# #     print apply_elementary_reflection(tau2, v2, R)
#      
#       
#     H.add_elementary_reflection(tau2, v2)
#     print H.apply_transpose(X)
#     H.remove_elementary_reflection()
#     R2 = H.apply(X2)
#     tau2a, v2a = compute_householder(R2[1,1], R2[2:,1])
#     H.add_elementary_reflection(tau2a, v2a)
#     print H.apply_transpose(X2)
# #     I = np.eye(m)[:,:n]
# #     print H.apply(I)
# #     print np.dot(H.apply(I).T, H.apply(I))
# #      
