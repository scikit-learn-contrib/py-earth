'''
Created on Feb 16, 2013

@author: jasonrudy
'''

import os
import numpy

from nose.tools import assert_true, assert_equal

from pyearth._forward import ForwardPasser
from pyearth._basis import (Basis, ConstantBasisFunction,
                            HingeBasisFunction, LinearBasisFunction)
from pyearth._types import BOOL

numpy.random.seed(0)
basis = Basis(10)
constant = ConstantBasisFunction()
basis.append(constant)
bf1 = HingeBasisFunction(constant, 0.1, 10, 1, False, 'x1')
bf2 = HingeBasisFunction(constant, 0.1, 10, 1, True, 'x1')
bf3 = LinearBasisFunction(bf1, 2, 'x2')
basis.append(bf1)
basis.append(bf2)
basis.append(bf3)
X = numpy.random.normal(size=(100, 10))
missing = numpy.zeros_like(X).astype(BOOL)
B = numpy.empty(shape=(100, 4), dtype=numpy.float64)
basis.transform(X, missing, B)
beta = numpy.random.normal(size=4)
y = numpy.empty(shape=100, dtype=numpy.float64)
y[:] = numpy.dot(B, beta) + numpy.random.normal(size=100)


def test_orthonormal_update():

    forwardPasser = ForwardPasser(X, missing, y[:, numpy.newaxis],
                                  numpy.ones(X.shape[0]),
                                  numpy.ones(1),
                                  max_terms=1000, penalty=1)

    numpy.set_printoptions(precision=4)
    m, n = X.shape
    B_orth = forwardPasser.get_B_orth()
    v = numpy.random.normal(size=m)
    for i in range(1, 10):
        v_ = numpy.random.normal(size=m)
        B_orth[:, i] = 10 * v_ + v
        v = v_
        forwardPasser.orthonormal_update(i)

        B_orth_dot_B_orth_T = numpy.dot(B_orth[:, 0:i + 1].transpose(),
                                        B_orth[:, 0:i + 1])
        assert_true(
            numpy.max(numpy.abs(
                B_orth_dot_B_orth_T - numpy.eye(i + 1))
            ) < .0000001
        )


def test_run():

    forwardPasser = ForwardPasser(X, missing, y[:, numpy.newaxis],
                                  numpy.ones(X.shape[0]),
                                  numpy.ones(1),
                                  max_terms=1000, penalty=1)

    forwardPasser.run()
    res = str(forwardPasser.get_basis()) + \
        '\n' + str(forwardPasser.trace())
    filename = os.path.join(os.path.dirname(__file__),
                            'forward_regress.txt')
    with open(filename, 'r') as fl:
        prev = fl.read()
    assert_equal(res, prev)
