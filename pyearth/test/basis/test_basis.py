import pickle
import numpy

from nose.tools import assert_equal, assert_true

from .base import BaseContainer
from pyearth._basis import (HingeBasisFunction, SmoothedHingeBasisFunction,
                            ConstantBasisFunction, LinearBasisFunction, Basis)


class Container(BaseContainer):

    def __init__(self):
        super(Container, self).__init__()
        self.basis = Basis(self.X.shape[1])
        self.parent = ConstantBasisFunction()
        self.bf1 = HingeBasisFunction(self.parent, 1.0, 10, 1, False)
        self.bf2 = HingeBasisFunction(self.parent, 1.0, 4, 2, True)
        self.bf3 = HingeBasisFunction(self.bf2, 1.0, 4, 3, True)
        self.bf4 = LinearBasisFunction(self.parent, 2)
        self.bf5 = HingeBasisFunction(self.parent, 1.5, 8, 2, True)
        self.basis.append(self.parent)
        self.basis.append(self.bf1)
        self.basis.append(self.bf2)
        self.basis.append(self.bf3)
        self.basis.append(self.bf4)
        self.basis.append(self.bf5)


def test_anova_decomp():
    cnt = Container()
    anova = cnt.basis.anova_decomp()
    assert_equal(set(anova[frozenset([1])]), set([cnt.bf1]))
    assert_equal(set(anova[frozenset([2])]), set([cnt.bf2, cnt.bf4,
                                                  cnt.bf5]))
    assert_equal(set(anova[frozenset([2, 3])]), set([cnt.bf3]))
    assert_equal(set(anova[frozenset()]), set([cnt.parent]))
    assert_equal(len(anova), 4)


def test_smooth_knots():
    cnt = Container()
    mins = [0.0, -1.0, 0.1, 0.2]
    maxes = [2.5, 3.5, 3.0, 2.0]
    knots = cnt.basis.smooth_knots(mins, maxes)
    assert_equal(knots[cnt.bf1], (0.0, 2.25))
    assert_equal(knots[cnt.bf2], (0.55, 1.25))
    assert_equal(knots[cnt.bf3], (0.6,  1.5))
    assert_true(cnt.bf4 not in knots)
    assert_equal(knots[cnt.bf5], (1.25, 2.25))


def test_smooth():
    cnt = Container()
    X = numpy.random.uniform(-2.0, 4.0, size=(20, 4))
    smooth_basis = cnt.basis.smooth(X)
    for bf, smooth_bf in zip(cnt.basis, smooth_basis):
        if type(bf) is HingeBasisFunction:
            assert_true(type(smooth_bf) is SmoothedHingeBasisFunction)
        elif type(bf) is ConstantBasisFunction:
            assert_true(type(smooth_bf) is ConstantBasisFunction)
        elif type(bf) is LinearBasisFunction:
            assert_true(type(smooth_bf) is LinearBasisFunction)
        else:
            raise AssertionError('Basis function is of an unexpected type.')
        assert_true(type(smooth_bf) in {SmoothedHingeBasisFunction,
                                        ConstantBasisFunction,
                                        LinearBasisFunction})
        if bf.has_knot():
            assert_equal(bf.get_knot(), smooth_bf.get_knot())


def test_add():
    cnt = Container()
    assert_equal(len(cnt.basis), 6)


def test_pickle_compat():
    cnt = Container()
    basis_copy = pickle.loads(pickle.dumps(cnt.basis))
    assert_true(cnt.basis == basis_copy)
