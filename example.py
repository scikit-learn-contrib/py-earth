'''
Created on Feb 19, 2013

@author: jasonrudy
'''
import numpy
import cProfile
#import pyearth._forward
#import _basis
from pyearth._forward import ForwardPasser
numpy.random.seed(0)
m = 1000
n = 10
X = 80*numpy.random.uniform(size=(m,n)) - 40
y = numpy.abs(X[:,6] - 4.0) + 1*numpy.random.normal(size=m)
#X -= numpy.mean(X,axis=0)
#X /= numpy.std(X,axis=0)
#from _basis import Basis
#X = numpy.random.normal(size=(100,10))
#y = numpy.random.normal(size=100)
#print numpy.std(X,axis=0)
#print numpy.mean(X,axis=0)
forwardPasser = ForwardPasser(X,y)
cProfile.run('forwardPasser.run()')

print forwardPasser.trace()
print forwardPasser.get_basis()