'''
Created on Feb 16, 2013

@author: jasonrudy
'''
import unittest
from pyearth._forward import ForwardPasser
from pyearth._basis import Basis, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
import numpy
        
class TestForwardPasser(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        self.basis = Basis()
        constant = ConstantBasisFunction()
        self.basis.append(constant)
        bf1 = HingeBasisFunction(constant,1.0,10,1,False,'x1')
        bf2 = HingeBasisFunction(constant,1.0,10,1,True,'x1')
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
        self.forwardPasser = ForwardPasser(self.X,self.y)
        
    def testRun(self):
        self.forwardPasser.run()
        res = str(self.forwardPasser.get_basis()) + '\n' + str(self.forwardPasser.trace())
#        with open('regress.txt','w') as fl:
#            fl.write(res)
        with open('regress.txt','r') as fl:
            prev = fl.read()
        self.assertEqual(res,prev)
        
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()