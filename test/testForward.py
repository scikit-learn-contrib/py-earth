'''
Created on Feb 16, 2013

@author: jasonrudy
'''
import unittest
from pyearth._forward import ForwardPassRecord, ForwardPassIteration, ForwardPasser
from pyearth._util import gcv
from pyearth._basis import Basis, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
import numpy

class TestForwardPassRecord(unittest.TestCase):

    def setUp(self):
        #Create a record
        num_samples = 1000
        num_variables = 10
        penalty = 3.0
        sst = 100.0
        self.record = ForwardPassRecord(num_samples, num_variables, penalty, sst)
        self.record.append(ForwardPassIteration(0, 3, -1.0, 63.0, 3, 0))
        self.record.append(ForwardPassIteration(0, 3, 1.3, 34.0, 5, 0))
        self.record.append(ForwardPassIteration(3, 6, 2.3, 18.0, 7, 0))

    def tearDown(self):
        pass

    def testStatistics(self):
        mses = [self.record.mse(i) for i in range(len(self.record))]
        mses_ = [self.record.iterations[i].mse for i in range(len(self.record))]
        gcvs = [self.record.gcv(i) for i in range(len(self.record))]
        gcvs_ = [gcv(self.record.iterations[i].mse, self.record.iterations[i].basis_size, self.record.num_samples, self.record.penalty) for i in range(len(self.record))]
        rsqs = [self.record.rsq(i) for i in range(len(self.record))]
        rsqs_ = [1 - (self.record.iterations[i].mse / self.record.sst) for i in range(len(self.record))]
        grsqs = [self.record.grsq(i) for i in range(len(self.record))]
        grsqs_ = [1 - (self.record.gcv(i) / gcv(self.record.sst, 1, self.record.num_samples, self.record.penalty)) for i in range(len(self.record))]
        self.assertListEqual(mses,mses_)
        self.assertListEqual(gcvs,gcvs_)
        self.assertListEqual(rsqs,rsqs_)
        self.assertListEqual(grsqs,grsqs_)
        
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
        self.B = numpy.empty(shape=(100,4))
        self.basis.transform(self.X,self.B)
        self.beta = numpy.random.normal(size=4)
        self.y = numpy.empty(shape=100,dtype=numpy.float64)
        self.y[:] = numpy.dot(self.B,self.beta) + numpy.random.normal(size=100)
        self.forwardPasser = ForwardPasser(self.X,self.y)
        
    def testRun(self):
        self.forwardPasser.run()
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()