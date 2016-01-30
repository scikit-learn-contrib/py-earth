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
    
    def push_from_column(self, alpha, x):
        s = np.dot(x, x)
        self.V[self.k + 1:, self.k] = x
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
            self.V[self.k + 1:, self.k] *= (1 / v_one)
        if self.k > 0:
            T_ = self.T[:self.k,:self.k]
            self.T[:self.k,self.k] = -tau * np.dot(np.dot(T_, self.V[:,:self.k].T), self.V[:,self.k])
            self.T[self.k, self.k] = tau
        else:
            self.T[0,0] = tau
        self.k += 1
        
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
    
    def reset(self):
        self.k = 0
        
    def apply(self, X):
        return X - np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k], np.dot(self.V[:,:self.k].T, X)))
    
    def apply_to_transpose(self, X):
        return X - (np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k], np.dot(self.V[:,:self.k].T, X.T)))).T
    
    def apply_transpose(self, X):
        return X - np.dot(self.V[:,:self.k], np.dot(self.T[:self.k,:self.k].T, np.dot(self.V[:,:self.k].T, X)))
    
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
        tau, v = compute_householder(x_[self.k], x_[(self.k + 1):])
        
        self.householder.push_elementary_reflection(tau, v)
        self.Q_t[self.k, self.k] = 1.
        self.Q_t[self.k, :] = self.householder.apply(self.Q_t[self.k, :])
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
    
