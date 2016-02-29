"""
=============================
Plotting simple sine function
=============================

A simple example plotting a fit of the sine function.
"""
import numpy
import matplotlib.pyplot as plt

from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
y = 100 * \
    (numpy.sin((X[:, 6])) - 4.0) + \
    10 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth(max_degree=3, minspan_alpha=.5, verbose=True)
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

# Plot the model
y_hat = model.predict(X)
plt.plot(X[:, 6], y, 'r.')
plt.plot(X[:, 6], y_hat, 'b.')
plt.show()
