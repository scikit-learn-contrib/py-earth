'''
Created on Feb 16, 2013

@author: jasonrudy
'''
import unittest
from _forward import ForwardPassRecord, ForwardPassIteration, ForwardPasser
from _util import gcv

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
    pass
    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()