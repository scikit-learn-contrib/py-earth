'''
Created on Feb 19, 2013

@author: jasonrudy
'''
import numpy
#import pyearth._forward
#import _basis
from pyearth._forward import ForwardPasser
#from _basis import Basis
X = numpy.random.normal(size=(100,10))
y = numpy.random.normal(size=100)
forwardPasser = ForwardPasser(X,y)