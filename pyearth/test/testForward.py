'''
Created on Feb 16, 2013

@author: jasonrudy
'''
import unittest
from _forward import ForwardPassRecord, ForwardPassIteration, ForwardPasser

class TestForwardPassRecord(unittest.TestCase):


    def setUp(self):
        #Create a record
        self.record = ForwardPassRecord()


    def tearDown(self):
        pass


    def testName(self):
        pass

class TestForwardPasser(unittest.TestCase):
    pass
    
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()