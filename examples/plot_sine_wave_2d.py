"""
==================================
Plotting two simple sine functions
==================================

A simple example plotting a fit of two sine functions.
"""
import numpy
import matplotlib.pyplot as plt

from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
y1 = 100 * \
    numpy.abs(numpy.sin((X[:, 6]) / 10) - 4.0) + \
    10 * numpy.random.normal(size=m)

y2 = 100 * \
    numpy.abs(numpy.sin((X[:, 6]) / 2) - 8.0) + \
    5 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth(max_degree=3, minspan_alpha=.5)
y_mix = numpy.concatenate((y1[:, numpy.newaxis], y2[:, numpy.newaxis]), axis=1)
model.fit(X, y_mix)

# Print the model
print(model.trace())
print(model.summary())

# Plot the model
y_hat = model.predict(X)

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
ax.plot(X[:, 6], y_mix[:, 0], 'r.')
ax.plot(X[:, 6], model.predict(X)[:, 0], 'b.')

ax = fig.add_subplot(1, 2, 2)
ax.plot(X[:, 6], y_mix[:, 1], 'r.')
ax.plot(X[:, 6], model.predict(X)[:, 1], 'b.')
plt.show()
