'''
Created on Jan 28, 2016

@author: jason
'''
import numpy as np
from pyearth._qr import UpdatingQT

def test_updating_qt():
    m = 10
    n = 3
    X = np.random.normal(size=(m, n))
    u = UpdatingQT(m, n)
    Q = np.linalg.qr(X, mode='reduced')[0]
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

if __name__ == '__main__':
    test_updating_qt()