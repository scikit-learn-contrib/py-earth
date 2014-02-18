'''
Created on Feb 17, 2013

@author: jasonrudy
'''
from nose.tools import assert_true, assert_false, assert_equal
from .._basis import Basis, ConstantBasisFunction, HingeBasisFunction, \
    LinearBasisFunction, SmoothedHingeBasisFunction
import numpy
import pickle
import os


class BaseTestClass(object):

    def __init__(self):
        numpy.random.seed(0)
        data = numpy.genfromtxt(
            os.path.join(os.path.dirname(__file__), 'test_data.csv'),
            delimiter=',', skip_header=1)
        self.y = numpy.array(data[:, 5])
        self.X = numpy.array(data[:, 0:5])


class TestConstantBasisFunction(BaseTestClass):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.bf = ConstantBasisFunction()

    def test_apply(self):
        m, n = self.X.shape
        B = numpy.empty(shape=(m, 10))

        assert_false(numpy.all(B[:, 0] == 1))
        self.bf.apply(self.X, B[:, 0])
        assert_true(numpy.all(B[:, 0] == 1))

    def test_pickle_compatibility(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf))
        assert_true(self.bf == bf_copy)
        
    def test_smoothed_version(self):
        smoothed = self.bf._smoothed_version(None, {}, {})
        assert_true(type(smoothed) is ConstantBasisFunction)

class TestSmoothedHingeBasisFunction(BaseTestClass):
    
    def __init__(self):
        super(self.__class__, self).__init__()
        self.parent = ConstantBasisFunction()
        self.bf1 = SmoothedHingeBasisFunction(self.parent, 1.0, 0.0, 3.0, 10, 1, False)
        self.bf2 = SmoothedHingeBasisFunction(self.parent, 1.0, 0.0, 3.0, 10, 1, True)
    
    def test_getters(self):
        assert not self.bf1.get_reverse()
        assert self.bf2.get_reverse()
        assert self.bf1.get_knot() == 1.0
        assert self.bf1.get_variable() == 1
        assert self.bf1.get_knot_idx() == 10
        assert self.bf1.get_parent() == self.parent
        assert self.bf1.get_knot_minus() == 0.0
        assert self.bf1.get_knot_plus() == 3.0
    
    def test_pickle_compatibility(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf1))
        assert_equal(self.bf1, bf_copy)
        
    def test_smoothed_version(self):
        translation = {self.parent: self.parent._smoothed_version(None, {}, {})}
        smoothed = self.bf1._smoothed_version(self.parent, {}, translation)
        assert_equal(self.bf1, smoothed)
        
    def test_degree(self):
        assert_equal(self.bf1.degree(), 1)
        assert_equal(self.bf2.degree(), 1)

    def test_p_r(self):
        pplus = (2*3.0 + 0.0 -3*1.0) / ((3.0 - 0.0)**2)
        rplus = (2*1.0 - 3.0 - 0.0) / ((3.0 - 0.0)**3)
        pminus = (3*1.0 - 2*0.0 - 3.0) / ((0.0 - 3.0)**2)
        rminus = (0.0 + 3.0 - 2*1.0) / ((0.0 - 3.0)**3)
        assert_equal(self.bf1.get_p(), pplus)
        assert_equal(self.bf1.get_r(), rplus)
        assert_equal(self.bf2.get_p(), pminus)
        assert_equal(self.bf2.get_r(), rminus)
        
    def test_apply(self):
        m, n = self.X.shape
        B = numpy.ones(shape=(m, 10))
        self.bf1.apply(self.X, B[:, 0])
        self.bf2.apply(self.X, B[:, 1])
        pplus = (2*3.0 + 0.0 -3*1.0) / ((3.0 - 0.0)**2)
        rplus = (2*1.0 - 3.0 - 0.0) / ((3.0 - 0.0)**3)
        pminus = (3*1.0 - 2*0.0 - 3.0) / ((0.0 - 3.0)**2)
        rminus = (0.0 + 3.0 - 2*1.0) / ((0.0 - 3.0)**3)
        c1 = numpy.ones(m)
        c1[self.X[:, 1] <= 0.0] = 0.0
        c1[(self.X[:, 1] > 0.0) & (self.X[:, 1] < 3.0)] = pplus*((self.X[(self.X[:, 1] > 0.0) & (self.X[:, 1] < 3.0), 1] - 0.0)**2) + \
                                                      rplus*((self.X[(self.X[:, 1] > 0.0) & (self.X[:, 1] < 3.0), 1] - 0.0)**3)
        c1[self.X[:,1] >= 3.0] = self.X[self.X[:,1] >= 3.0, 1] - 1.0
        c2 = numpy.ones(m)
        c2[self.X[:, 1] >= 3.0] = 0.0
        c2[self.X[:, 1] <= 0.0] = -1 * (self.X[self.X[:, 1] <= 0.0] - 1.0)
        c2[(self.X[:, 1] > 0.0) & (self.X[:, 1] < 3.0)] = pminus*((self.X[(self.X[:, 1] > 0.0) & (self.X[:, 1] < 3.0), 1] - 3.0)**2) + \
                                                          rminus*((self.X[(self.X[:, 1] > 0.0) & (self.X[:, 1] < 3.0), 1] - 3.0)**3)
        assert_true(
            numpy.all(numpy.abs(B[:, 0] - c1) < .0000001))
        assert_true(
            numpy.all(numpy.abs(B[:, 1] - c2) < .0000001))
        
class TestHingeBasisFunction(BaseTestClass):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.parent = ConstantBasisFunction()
        self.bf = HingeBasisFunction(self.parent, 1.0, 10, 1, False)

    def test_getters(self):
        assert not self.bf.get_reverse()
        assert self.bf.get_knot() == 1.0
        assert self.bf.get_variable() == 1
        assert self.bf.get_knot_idx() == 10
        assert self.bf.get_parent() == self.parent

    def test_apply(self):
        m, n = self.X.shape
        B = numpy.ones(shape=(m, 10))
        self.bf.apply(self.X, B[:, 0])
        assert_true(
            numpy.all(B[:, 0] == (self.X[:, 1] - 1.0) * (self.X[:, 1] > 1.0)))

    def test_degree(self):
        assert_equal(self.bf.degree(), 1)

    def test_pickle_compatibility(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf))
        assert_true(self.bf == bf_copy)

    def test_smoothed_version(self):
        knot_dict = {self.bf: (.5,1.5)}
        translation = {self.parent: self.parent._smoothed_version(None, {}, {})}
        smoothed = self.bf._smoothed_version(self.parent, knot_dict, translation)
        assert_true(type(smoothed) is SmoothedHingeBasisFunction)
        assert_true(translation[self.parent] is smoothed.get_parent())
        assert_equal(smoothed.get_knot_minus(), 0.5)
        assert_equal(smoothed.get_knot_plus(), 1.5)
        assert_equal(smoothed.get_knot(), self.bf.get_knot())
        assert_equal(smoothed.get_variable(), self.bf.get_variable())

class TestLinearBasisFunction(BaseTestClass):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.parent = ConstantBasisFunction()
        self.bf = LinearBasisFunction(self.parent, 1)

    def test_apply(self):
        m, n = self.X.shape
        B = numpy.ones(shape=(m, 10))
        self.bf.apply(self.X, B[:, 0])
        assert_true(numpy.all(B[:, 0] == self.X[:, 1]))

    def test_degree(self):
        assert_equal(self.bf.degree(), 1)

    def test_pickle_compatibility(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf))
        assert_true(self.bf == bf_copy)
    
    def test_smoothed_version(self):
        translation = {self.parent: self.parent._smoothed_version(None, {}, {})}
        smoothed = self.bf._smoothed_version(self.parent, {}, translation)
        assert_true(type(smoothed) is LinearBasisFunction)
        assert_equal(smoothed.get_variable(), self.bf.get_variable())
        
class TestBasis(BaseTestClass):

    def __init__(self):
        super(self.__class__, self).__init__()
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

    def test_anova_decomp(self):
        anova = self.basis.anova_decomp()
        assert_equal(set(anova[frozenset([1])]), set([self.bf1]))
        assert_equal(set(anova[frozenset([2])]), set([self.bf2, self.bf4, self.bf5]))
        assert_equal(set(anova[frozenset([2, 3])]), set([self.bf3]))
        assert_equal(set(anova[frozenset()]), set([self.parent]))
        assert_equal(len(anova), 4)
        
    def test_smooth_knots(self):
        mins = [0.0, -1.0, 0.1, 0.2]
        maxes = [2.5, 3.5, 3.0, 2.0]
        knots = self.basis.smooth_knots(mins, maxes)
        assert_equal(knots[self.bf1], (0.0, 2.25))
        assert_equal(knots[self.bf2], (0.55, 1.25))
        assert_equal(knots[self.bf3], (0.6,  1.5))
        assert_true(self.bf4 not in knots)
        assert_equal(knots[self.bf5], (1.25, 2.25))
        
    def test_smooth(self):
        X = numpy.random.uniform(-2.0, 4.0, size=(20,4))
        smooth_basis = self.basis.smooth(X)
        for bf, smooth_bf in zip(self.basis, smooth_basis):
            if type(bf) is HingeBasisFunction:
                assert_true(type(smooth_bf) is SmoothedHingeBasisFunction)
            elif type(bf) is ConstantBasisFunction:
                assert_true(type(smooth_bf) is ConstantBasisFunction)
            elif type(bf) is LinearBasisFunction:
                assert_true(type(smooth_bf) is LinearBasisFunction)
            else:
                raise AssertionError('Basis function is of an unexpected type.')
            assert_true(type(smooth_bf) in {SmoothedHingeBasisFunction, ConstantBasisFunction, LinearBasisFunction})
            if bf.has_knot():
                assert_equal(bf.get_knot(), smooth_bf.get_knot())
        
    def test_add(self):
        assert_equal(len(self.basis), 6)

    def test_pickle_compat(self):
        basis_copy = pickle.loads(pickle.dumps(self.basis))
        assert_true(self.basis == basis_copy)

if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])
