import pickle
import numpy

from nose.tools import assert_equal, assert_true

from .base import BaseContainer
from pyearth._types import BOOL
from pyearth._basis import LinearBasisFunction, ConstantBasisFunction


class Container(BaseContainer):

    def __init__(self):
        super(Container, self).__init__()
        self.parent = ConstantBasisFunction()
        self.bf = LinearBasisFunction(self.parent, 1)


def test_apply():
    cnt = Container()
    m, n = cnt.X.shape
    missing = numpy.zeros_like(cnt.X, dtype=BOOL)
    B = numpy.ones(shape=(m, 10))
    cnt.bf.apply(cnt.X, missing, B[:, 0])
    assert_true(numpy.all(B[:, 0] == cnt.X[:, 1]))


def test_apply_deriv():
    cnt = Container()
    m, _ = cnt.X.shape
    missing = numpy.zeros_like(cnt.X, dtype=BOOL)
    b = numpy.empty(shape=m)
    j = numpy.empty(shape=m)
    cnt.bf.apply_deriv(cnt.X, missing, b, j, 1)
    numpy.testing.assert_almost_equal(b, cnt.X[:, 1])
    numpy.testing.assert_almost_equal(j, 1.0)


def test_degree():
    cnt = Container()
    assert_equal(cnt.bf.degree(), 1)


def test_pickle_compatibility():
    cnt = Container()
    bf_copy = pickle.loads(pickle.dumps(cnt.bf))
    assert_true(cnt.bf == bf_copy)


def test_smoothed_version():
    cnt = Container()
    translation = {cnt.parent: cnt.parent._smoothed_version(None, {}, {})}
    smoothed = cnt.bf._smoothed_version(cnt.parent, {}, translation)
    assert_true(isinstance(smoothed, LinearBasisFunction))
    assert_equal(smoothed.get_variable(), cnt.bf.get_variable())
