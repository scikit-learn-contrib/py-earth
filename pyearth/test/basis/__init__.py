import pickle
import os
from nose.tools import assert_true, assert_false, assert_equal

import numpy

from pyearth._basis import Basis, ConstantBasisFunction, HingeBasisFunction, \
    LinearBasisFunction, SmoothedHingeBasisFunction

numpy.random.seed(0)
