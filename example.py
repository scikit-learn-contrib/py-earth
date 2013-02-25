
import numpy
import cProfile
#import pyearth._forward
#import _basis
from pyearth import Earth

#Create some fake data
numpy.random.seed(0)
m = 1000
n = 10
X = 80*numpy.random.uniform(size=(m,n)) - 40
y = numpy.abs(X[:,6] - 4.0) + 1*numpy.random.normal(size=m)

#Fit an Earth model
model = Earth()
cProfile.run('model.fit(X,y)')


print model.trace()
print model