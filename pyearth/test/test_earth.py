'''
Created on Feb 24, 2013

@author: jasonrudy
'''
import numpy
from pyearth._basis import Basis, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
from pyearth import Earth
import pickle
import copy
import os
from testing_utils import if_pandas, if_patsy
from nose.tools import assert_equal, assert_not_equal, assert_true, assert_false, \
    assert_almost_equal, assert_list_equal
    
class TestEarth(object):

    def __init__(self):
        numpy.random.seed(0)
        self.basis = Basis()
        constant = ConstantBasisFunction()
        self.basis.append(constant)
        bf1 = HingeBasisFunction(constant,0.1,10,1,False,'x1')
        bf2 = HingeBasisFunction(constant,0.1,10,1,True,'x1')
        bf3 = LinearBasisFunction(bf1,2,'x2')
        self.basis.append(bf1)
        self.basis.append(bf2)
        self.basis.append(bf3)
        self.X = numpy.random.normal(size=(100,10))
        self.B = numpy.empty(shape=(100,4),dtype=numpy.float64)
        self.basis.transform(self.X,self.B)
        self.beta = numpy.random.normal(size=4)
        self.y = numpy.empty(shape=100,dtype=numpy.float64)
        self.y[:] = numpy.dot(self.B,self.beta) + numpy.random.normal(size=100)
        self.earth = Earth(penalty=1)

    def test_fit(self):
        self.earth.fit(self.X, self.y)
        res = str(self.earth.trace()) + '\n' + str(self.earth)
#        with open('earth_regress.txt','w') as fl:
#            fl.write(res)
        with open(os.path.join(os.path.dirname(__file__),'earth_regress.txt'),'r') as fl:
            prev = fl.read()
        assert_equal(res,prev)
        
    def test_score(self):
        model = self.earth.fit(self.X, self.y)
        record = model.pruning_trace()
        grsq = record.grsq(record.get_selected())
        assert_almost_equal(grsq,model.score(self.X,self.y))

    @if_pandas
    def test_pandas_compatibility(self):
        import pandas
        X = pandas.DataFrame(self.X)
        y = pandas.DataFrame(self.y)
        colnames = ['xx'+str(i) for i in range(X.shape[1])]
        X.columns = colnames
        model = self.earth.fit(X,y)
        assert_list_equal(colnames,model.xlabels)
        
    @if_patsy
    @if_pandas
    def test_patsy_compatibility(self):
        import pandas
        import patsy
        X = pandas.DataFrame(self.X)
        y = pandas.DataFrame(self.y)
        colnames = ['xx'+str(i) for i in range(X.shape[1])]
        X.columns = colnames
        X['y'] = y
        y, X = patsy.dmatrices('y ~ xx0 + xx1 + xx2 + xx3 + xx4 + xx5 + xx6 + xx7 + xx8 + xx9 - 1',data=X)
        model = self.earth.fit(X,y)
        assert_list_equal(colnames,model.xlabels)
        
    def test_pickle_compatibility(self):
        model = self.earth.fit(self.X, self.y)
        model_copy = pickle.loads(pickle.dumps(model))
        assert_true(model_copy == model)
        assert_true(numpy.all(model.predict(self.X) == model_copy.predict(self.X)))
        assert_true(model.basis_[0] is model.basis_[1]._get_root())
        assert_true(model_copy.basis_[0] is model_copy.basis_[1]._get_root())
        
    def test_copy_compatibility(self):
        model = self.earth.fit(self.X, self.y)
        model_copy = copy.copy(model)
        assert_true(model_copy == model)
        assert_true(numpy.all(model.predict(self.X) == model_copy.predict(self.X)))
        assert_true(model.basis_[0] is model.basis_[1]._get_root())
        assert_true(model_copy.basis_[0] is model_copy.basis_[1]._get_root())
        
if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])
