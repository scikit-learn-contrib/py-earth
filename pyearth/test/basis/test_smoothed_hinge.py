import pickle
import numpy

from nose.tools import assert_equal

from .base import BaseContainer
from pyearth._types import BOOL
from pyearth._basis import SmoothedHingeBasisFunction, ConstantBasisFunction


class Container(BaseContainer):

    def __init__(self):
        super(Container, self).__init__()
        self.parent = ConstantBasisFunction()
        self.bf1 = SmoothedHingeBasisFunction(self.parent,
                                              1.0, 0.0, 3.0, 10, 1,
                                              False)
        self.bf2 = SmoothedHingeBasisFunction(self.parent,
                                              1.0, 0.0, 3.0, 10, 1,
                                              True)


def test_getters():
    cnt = Container()
    assert not cnt.bf1.get_reverse()
    assert cnt.bf2.get_reverse()
    assert cnt.bf1.get_knot() == 1.0
    assert cnt.bf1.get_variable() == 1
    assert cnt.bf1.get_knot_idx() == 10
    assert cnt.bf1.get_parent() == cnt.parent
    assert cnt.bf1.get_knot_minus() == 0.0
    assert cnt.bf1.get_knot_plus() == 3.0


def test_pickle_compatibility():
    cnt = Container()
    bf_copy = pickle.loads(pickle.dumps(cnt.bf1))
    assert_equal(cnt.bf1, bf_copy)


def test_smoothed_version():
    cnt = Container()
    translation = {cnt.parent: cnt.parent._smoothed_version(None, {}, {})}
    smoothed = cnt.bf1._smoothed_version(cnt.parent, {}, translation)
    assert_equal(cnt.bf1, smoothed)


def test_degree():
    cnt = Container()
    assert_equal(cnt.bf1.degree(), 1)
    assert_equal(cnt.bf2.degree(), 1)


def test_p_r():
    cnt = Container()
    pplus = (2 * 3.0 + 0.0 - 3 * 1.0) / ((3.0 - 0.0)**2)
    rplus = (2 * 1.0 - 3.0 - 0.0) / ((3.0 - 0.0)**3)
    pminus = (3 * 1.0 - 2 * 0.0 - 3.0) / ((0.0 - 3.0)**2)
    rminus = (0.0 + 3.0 - 2 * 1.0) / ((0.0 - 3.0)**3)
    assert_equal(cnt.bf1.get_p(), pplus)
    assert_equal(cnt.bf1.get_r(), rplus)
    assert_equal(cnt.bf2.get_p(), pminus)
    assert_equal(cnt.bf2.get_r(), rminus)


def test_apply():
    cnt = Container()
    m, _ = cnt.X.shape
    missing = numpy.zeros_like(cnt.X, dtype=BOOL)
    B = numpy.ones(shape=(m, 10))
    cnt.bf1.apply(cnt.X, missing, B[:, 0])
    cnt.bf2.apply(cnt.X, missing, B[:, 1])
    pplus = (2 * 3.0 + 0.0 - 3 * 1.0) / ((3.0 - 0.0)**2)
    rplus = (2 * 1.0 - 3.0 - 0.0) / ((3.0 - 0.0)**3)
    pminus = (3 * 1.0 - 2 * 0.0 - 3.0) / ((0.0 - 3.0)**2)
    rminus = (0.0 + 3.0 - 2 * 1.0) / ((0.0 - 3.0)**3)
    c1 = numpy.ones(m)
    c1[cnt.X[:, 1] <= 0.0] = 0.0
    c1[(cnt.X[:, 1] > 0.0) & (cnt.X[:, 1] < 3.0)] = (
        pplus * ((cnt.X[(cnt.X[:, 1] > 0.0) & (
            cnt.X[:, 1] < 3.0), 1] - 0.0)**2) +
        rplus * ((cnt.X[(cnt.X[:, 1] > 0.0) & (
            cnt.X[:, 1] < 3.0), 1] - 0.0)**3))
    c1[cnt.X[:, 1] >= 3.0] = cnt.X[cnt.X[:, 1] >= 3.0, 1] - 1.0
    c2 = numpy.ones(m)
    c2[cnt.X[:, 1] >= 3.0] = 0.0
    c2.flat[cnt.X[:, 1] <= 0.0] = -1 * (cnt.X[cnt.X[:, 1] <= 0.0] - 1.0)
    c2[(cnt.X[:, 1] > 0.0) & (cnt.X[:, 1] < 3.0)] = (
        pminus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                         (cnt.X[:, 1] < 3.0), 1] - 3.0)**2) +
        rminus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                         (cnt.X[:, 1] < 3.0), 1] - 3.0)**3)
    )
    numpy.testing.assert_almost_equal(B[:, 0], c1)
    numpy.testing.assert_almost_equal(B[:, 1], c2)


def test_apply_deriv():
    cnt = Container()
    m, _ = cnt.X.shape
    missing = numpy.zeros_like(cnt.X, dtype=BOOL)
    pplus = (2 * 3.0 + 0.0 - 3 * 1.0) / ((3.0 - 0.0)**2)
    rplus = (2 * 1.0 - 3.0 - 0.0) / ((3.0 - 0.0)**3)
    pminus = (3 * 1.0 - 2 * 0.0 - 3.0) / ((0.0 - 3.0)**2)
    rminus = (0.0 + 3.0 - 2 * 1.0) / ((0.0 - 3.0)**3)
    c1 = numpy.ones(m)
    c1[cnt.X[:, 1] <= 0.0] = 0.0
    c1[(cnt.X[:, 1] > 0.0) & (cnt.X[:, 1] < 3.0)] = (
        pplus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                        (cnt.X[:, 1] < 3.0), 1] - 0.0)**2) +
        rplus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                        (cnt.X[:, 1] < 3.0), 1] - 0.0)**3))
    c1[cnt.X[:, 1] >= 3.0] = cnt.X[cnt.X[:, 1] >= 3.0, 1] - 1.0
    c2 = numpy.ones(m)
    c2[cnt.X[:, 1] >= 3.0] = 0.0
    c2.flat[cnt.X[:, 1] <= 0.0] = -1 * (cnt.X[cnt.X[:, 1] <= 0.0] - 1.0)
    c2[(cnt.X[:, 1] > 0.0) & (cnt.X[:, 1] < 3.0)] = (
        pminus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                         (cnt.X[:, 1] < 3.0), 1] - 3.0)**2) +
        rminus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                         (cnt.X[:, 1] < 3.0), 1] - 3.0)**3)
    )
    b1 = numpy.empty(shape=m)
    j1 = numpy.empty(shape=m)
    b2 = numpy.empty(shape=m)
    j2 = numpy.empty(shape=m)
    cp1 = numpy.ones(m)
    cp1[cnt.X[:, 1] <= 0.0] = 0.0
    cp1[(cnt.X[:, 1] > 0.0) & (cnt.X[:, 1] < 3.0)] = (
        2.0 * pplus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                              (cnt.X[:, 1] < 3.0), 1] - 0.0)) +
        3.0 * rplus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                              (cnt.X[:, 1] < 3.0), 1] - 0.0)**2)
    )
    cp1[cnt.X[:, 1] >= 3.0] = 1.0
    cp2 = numpy.ones(m)
    cp2[cnt.X[:, 1] >= 3.0] = 0.0
    cp2[cnt.X[:, 1] <= 0.0] = -1.0
    cp2[(cnt.X[:, 1] > 0.0) & (cnt.X[:, 1] < 3.0)] = (
        2.0 * pminus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                               (cnt.X[:, 1] < 3.0), 1] - 3.0)) +
        3.0 * rminus * ((cnt.X[(cnt.X[:, 1] > 0.0) &
                               (cnt.X[:, 1] < 3.0), 1] - 3.0)**2))
    cnt.bf1.apply_deriv(cnt.X, missing, b1, j1, 1)
    cnt.bf2.apply_deriv(cnt.X, missing, b2, j2, 1)
    numpy.testing.assert_almost_equal(b1, c1)
    numpy.testing.assert_almost_equal(b2, c2)
    numpy.testing.assert_almost_equal(j1, cp1)
    numpy.testing.assert_almost_equal(j2, cp2)
