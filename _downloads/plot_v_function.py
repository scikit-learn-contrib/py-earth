"""
====================================
Plotting the absolute value function
====================================

A simple example plotting a fit of the absolute value function.

"""

import numpy
import matplotlib.pyplot as plt
from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 1000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
y = numpy.abs(X[:, 6] - 4.0) + 1 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth(max_degree=1, verbose=True)
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

# Plot the model
y_hat = model.predict(X)
plt.figure()
plt.plot(X[:, 6], y, 'r.')
plt.plot(X[:, 6], y_hat, 'b.')
plt.show()
