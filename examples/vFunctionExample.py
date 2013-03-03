
import numpy
import cProfile
from pyearth import Earth
from matplotlib import pyplot
numpy.set_printoptions(precision=4)
#Create some fake data
numpy.random.seed(2)
m = 1000
n = 10
X = 80*numpy.random.uniform(size=(m,n)) - 40
y = numpy.abs(X[:,6] - 4.0)**2 + 10*numpy.random.normal(size=m)
X -= X.mean(axis=0)
X /= X.std(axis=0)
y -= y.mean()
y /= y.std()

#Fit an Earth model
model = Earth(max_degree = 2, minspan=5)
cProfile.run('model.fit(X,y)')
#for bf in model.basis:
#    bf.unprune()
#model.linear_fit(X,y)

#Print the model
print model.trace()
print model

#Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X[:,6],y,'r.')
pyplot.plot(X[:,6],y_hat,'b.')
pyplot.show()

