import pickle
import numpy

from nose.tools import assert_true, assert_false

from .base import BaseContainer
from pyearth._types import BOOL
from pyearth._basis import ConstantBasisFunction


class Container(BaseContainer):

    def __init__(self):
        super(Container, self).__init__()
        self.bf = ConstantBasisFunction()


def test_apply():
    cnt = Container()
    m, _ = cnt.X.shape
    missing = numpy.zeros_like(cnt.X, dtype=BOOL)
    B = numpy.empty(shape=(m, 10))
    assert_false(numpy.all(B[:, 0] == 1))
    cnt.bf.apply(cnt.X, missing, B[:, 0])
    assert_true(numpy.all(B[:, 0] == 1))


def test_deriv():
    cnt = Container()
    m, _ = cnt.X.shape
    missing = numpy.zeros_like(cnt.X, dtype=BOOL)
    b = numpy.empty(shape=m)
    j = numpy.empty(shape=m)
    cnt.bf.apply_deriv(cnt.X, missing, b, j, 1)
    assert_true(numpy.all(b == 1))
    assert_true(numpy.all(j == 0))


def test_pickle_compatibility():
    cnt = Container()
    bf_copy = pickle.loads(pickle.dumps(cnt.bf))
    assert_true(cnt.bf == bf_copy)


def test_smoothed_version():
    cnt = Container()
    smoothed = cnt.bf._smoothed_version(None, {}, {})
    assert_true(type(smoothed) is ConstantBasisFunction)
