import matplotlib as mpl
mpl.use('Agg')
import numpy
from pyearth import Earth
from matplotlib import pyplot

#=========================================================================
# V-Function Example
#=========================================================================
# Create some fake data
numpy.random.seed(0)
m = 1000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
y = numpy.abs(X[:, 6] - 4.0) + 1 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth()
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

# Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X[:, 6], y, 'r.')
pyplot.plot(X[:, 6], y_hat, 'b.')
pyplot.xlabel('x_6')
pyplot.ylabel('y')
pyplot.title('Simple Earth Example')
pyplot.savefig('simple_earth_example.png')

#=========================================================================
# Hinge plot
#=========================================================================
from xkcdify import XKCDify
x = numpy.arange(-10, 10, .1)
y = x * (x > 0)

fig = pyplot.figure(figsize=(10, 5))
pyplot.plot(x, y)
ax = pyplot.gca()

pyplot.title('Basic Hinge Function')
pyplot.xlabel('x')
pyplot.ylabel('h(x)')
pyplot.annotate('x=t', (0, 0), xytext=(-30, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
XKCDify(ax)
pyplot.setp(ax, frame_on=False)
pyplot.savefig('hinge.png')

#=========================================================================
# Piecewise Linear Plot
#=========================================================================
m = 1000
x = numpy.arange(-10, 10, .1)
y = 1 - 2 * (1 - x) * (x < 1) + 0.5 * (x - 1) * (x > 1)

pyplot.figure(figsize=(10, 5))
pyplot.plot(x, y)
ax = pyplot.gca()
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.title('Piecewise Linear Function')
XKCDify(ax)
pyplot.setp(ax, frame_on=False)
pyplot.savefig('piecewise_linear.png')
