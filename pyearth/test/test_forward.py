'''
Created on Feb 16, 2013

@author: jasonrudy
'''

import os
import numpy

from nose.tools import assert_equal

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
sample_weight = numpy.ones((X.shape[0], 1))


def test_run():

    forwardPasser = ForwardPasser(X, missing, y[:, numpy.newaxis],
                                  sample_weight,
                                  max_terms=1000, penalty=1)

    forwardPasser.run()
    res = str(forwardPasser.get_basis()) + \
        '\n' + str(forwardPasser.trace())
    filename = os.path.join(os.path.dirname(__file__),
                            'forward_regress.txt')
#     with open(filename, 'w') as fl:
#         fl.write(res)
    with open(filename, 'r') as fl:
        prev = fl.read()
    assert_equal(res, prev)
