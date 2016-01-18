from pyearth._knot_search import OutcomeDependentData
from nose.tools import assert_true, assert_equal
import numpy as np
from numpy.testing.utils import assert_almost_equal
from scipy.linalg import qr

def test_outcome_dependent_data():
    m = 1000
    max_terms = 100
    y = np.random.normal(size=m)
    w = np.random.normal(size=m) ** 2
    data = OutcomeDependentData.alloc(y, w, m, max_terms)
    
    # Test updating
    B = np.empty(shape=(m,max_terms))
    for k in range(max_terms):
        b = np.random.normal(size=m)
        B[:,k] = b
        code = data.update(b, 1e-16)
        assert_equal(code, 0)
        assert_almost_equal(np.dot(data.Q_t[:k+1,:], np.transpose(data.Q_t[:k+1,:])),
                            np.eye(k+1))
    assert_equal(data.update(b, 1e-16), -1)
    
    # Test downdating
    q = np.array(data.Q_t).copy()
    theta = np.array(data.theta[:max_terms]).copy()
    data.downdate()
    data.update(b, 1e-16)
    assert_almost_equal(q, np.array(data.Q_t))
    assert_almost_equal(theta, np.array(data.theta[:max_terms]))
    assert_almost_equal(np.array(data.theta[:max_terms]), np.dot(data.Q_t, y))
    wB = B * w[:,None]
    Q, _ = qr(wB, pivoting=False, mode='economic')
    assert_almost_equal(np.abs(np.dot(data.Q_t, Q)), np.eye(max_terms))
    
    # Test that reweighting works
    w2 = np.random.normal(size=m) ** 2
    data.reweight(w2, B, 1e-16)
    w2B = B * w2[:,None]
    Q2, _ = qr(w2B, pivoting=False, mode='economic')
    assert_almost_equal(np.abs(np.dot(data.Q_t, Q2)), np.eye(max_terms))
    assert_almost_equal(np.array(data.theta[:max_terms]), np.dot(data.Q_t, y))
    
    
    





