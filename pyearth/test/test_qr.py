'''
Created on Jan 28, 2016

@author: jason
'''
import numpy as np
from pyearth._qr import UpdatingQT


def test_updating_qt():
    np.random.seed(0)
    m = 10
    n = 3

    X = np.random.normal(size=(n, m)).T
    u = UpdatingQT.alloc(m, n, 1e-14)

#     u2 = UpdatingQT(m, n)
    Q = np.linalg.qr(X, mode='reduced')[0]
    u.update(X[:, 0])

#     u2.update(X[:,0])
    u.update(X[:, 1])
#     u2.update(X[:,1])
    u.update(X[:, 2])
#     u2.update(X[:,2])

#     assert np.max(np.abs(np.abs(u2.Q_t) - np.abs(Q.T))) < .0000000000001
    assert np.max(np.abs(np.abs(u.Q_t) - np.abs(Q.T))) < .0000000000001

    X2 = X.copy()
    X2[:, 2] = np.random.normal(size=m)

    u.downdate()
    u.update(X2[:, 2])

    Q2 = np.linalg.qr(X2, mode='reduced')[0]
    assert np.max(np.abs(np.abs(u.Q_t) - np.abs(Q2.T))) < .0000000000001


def test_updating_qr_with_linear_dependence():
    np.random.seed(0)
    m = 10
    n = 5
    assert n >= 3
    X = np.random.normal(size=(n, m)).T
    X[:, 2] = X[:, 0] + 3 * X[:, 1]

    Q = np.linalg.qr(X, mode='reduced')[0]
    u = UpdatingQT.alloc(m, n, 1e-14)
    u2 = UpdatingQT.alloc(m, n, 1e-14)

    u.update(X[:, 0])
    u2.update(X[:, 0])
#     u2.update(X[:,0])
    u.update(X[:, 1])
    u2.update(X[:, 1])
#     u2.update(X[:,1])
    u.update(X[:, 2])

    assert np.max(
        np.abs(np.abs(u.Q_t[:2, :]) - np.abs(Q[:, :2].T))) < .0000000000001
    assert np.max(np.abs(u.Q_t[2, :])) == 0.

    # Make sure you can downdate a dependent column safely
    u.downdate()
    u.update(X[:, 2])
    assert np.max(
        np.abs(np.abs(u.Q_t[:2, :]) - np.abs(Q[:, :2].T))) < .0000000000001
    assert np.max(np.abs(u.Q_t[2, :])) == 0.

    for j in range(3, n):
        u.update(X[:, j])
        u2.update(X[:, j])

    # Q_t is orthonormal except for its zero column
    Q_nonzero = np.concatenate([u.Q_t[:2, :].T, u.Q_t[3:, :].T], axis=1)
    np.testing.assert_array_almost_equal(
        np.dot(Q_nonzero.T, Q_nonzero), np.eye(n - 1))

    # Q_t.T is in the column space of X
    b = np.linalg.lstsq(X, u.Q_t.T)[0]
    Q_hat = np.dot(X, b)
    np.testing.assert_array_almost_equal(Q_hat, u.Q_t.T)

    # X is in the column space of Q_t.T
    a = np.linalg.lstsq(u.Q_t.T, X)[0]
    X_hat = np.dot(u.Q_t.T, a)
    np.testing.assert_array_almost_equal(X_hat, X)

    # u and u2 should have the same householder
    np.testing.assert_array_almost_equal(
        u.householder.V[:, :u.householder.k],
        u2.householder.V[:, :u2.householder.k])
    np.testing.assert_array_almost_equal(
            u.householder.T[:u.householder.k, :u.householder.k],
            u2.householder.T[:u2.householder.k, :u2.householder.k])

    # u should have one more column than u2
    assert u.k == u2.k + 1
