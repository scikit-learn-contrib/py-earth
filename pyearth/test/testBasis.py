'''
Created on Feb 17, 2013

@author: jasonrudy
'''
import unittest
from pyearth._basis import Basis, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
import numpy
import pandas
import pickle

class Test(unittest.TestCase):


    def setUp(self):
        numpy.random.seed(0)
        self.data = pandas.read_csv('testData.csv')
        self.y = numpy.array(self.data['y'])
        self.X = numpy.array(self.data[['x1','x2','x3','x4']])
        


    def tearDown(self):
        pass


    def testName(self):
        pass


class TestConstantBasisFunction(Test):
        
    def setUp(self):
        super(self.__class__,self).setUp()
        self.bf = ConstantBasisFunction()
        
    def testApply(self):
        m,n = self.X.shape
        B = numpy.empty(shape=(m,10))
        
        self.assertFalse(numpy.all(B[:,0] == 1))
        self.bf.apply(self.X,B[:,0])
        self.assertTrue(numpy.all(B[:,0] == 1))
        
    def testPickleCompat(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf))
        self.assertTrue(self.bf == bf_copy)

class TestHingeBasisFunction(Test):
    def setUp(self):
        super(self.__class__,self).setUp()
        parent = ConstantBasisFunction()
        self.bf = HingeBasisFunction(parent,1.0,10,1,False)
        
    def testApply(self):
        m,n = self.X.shape
        B = numpy.ones(shape=(m,10))
        self.bf.apply(self.X,B[:,0])
        self.assertTrue(numpy.all(B[:,0] == (self.X[:,1] - 1.0) * (self.X[:,1] > 1.0)))
    
    def testDegree(self):
        self.assertEqual(self.bf.degree(),1)
        
    def testPickleCompat(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf))
        self.assertTrue(self.bf == bf_copy)
        
class TestLinearBasisFunction(Test):
    def setUp(self):
        super(self.__class__,self).setUp()
        parent = ConstantBasisFunction()
        self.bf = LinearBasisFunction(parent,1)
        
    def testApply(self):
        m,n = self.X.shape
        B = numpy.ones(shape=(m,10))
        self.bf.apply(self.X,B[:,0])
        self.assertTrue(numpy.all(B[:,0] == self.X[:,1]))
    
    def testDegree(self):
        self.assertEqual(self.bf.degree(),1)
        
    def testPickleCompat(self):
        bf_copy = pickle.loads(pickle.dumps(self.bf))
        self.assertTrue(self.bf == bf_copy)
    
class TestBasis(Test):
    def setUp(self):
        super(self.__class__,self).setUp()
        self.basis = Basis()
        self.parent = ConstantBasisFunction()
        self.bf = HingeBasisFunction(self.parent,1.0,10,1,False)
        self.basis.append(self.parent)
        self.basis.append(self.bf)
        
    def testAdd(self):
        self.assertEqual(len(self.basis),2)
    
    def testTranslateAndScale(self):
        m,n = self.X.shape
        numpy.random.seed(1)
        B = numpy.empty(shape=(m,self.basis.plen()))
        self.basis.transform(self.X,B)
        B_ = numpy.empty(shape=(m,self.basis.plen()))
        mu = numpy.mean(self.X,axis=0)
        sigma = numpy.std(self.X,axis=0)
        coeff = numpy.random.normal(size=B.shape[1])
        X_ = self.X * sigma + mu
        coeff_ = coeff.copy()
        self.basis.translate(sigma,mu)
        self.basis.scale(sigma,mu,coeff_)
        self.basis.transform(X_,B_)
        self.assertTrue(numpy.all((numpy.dot(B,coeff) - numpy.dot(B_,coeff_))**2 < 1e-12))
    
    def testPickleCompat(self):
        basis_copy = pickle.loads(pickle.dumps(self.basis))
        self.assertTrue(self.basis == basis_copy)
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()