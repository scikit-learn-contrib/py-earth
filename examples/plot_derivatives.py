"""
============================================
Plotting derivatives of simple sine function
============================================

A simple example plotting a fit of the sine function and
the derivatives computed by Earth.
"""
import numpy
import matplotlib.pyplot as plt

from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10
X = 20 * numpy.random.uniform(size=(m, n)) - 10
y = 10*numpy.sin(X[:, 6]) + 0.25*numpy.random.normal(size=m)

# Compute the known true derivative with respect to the predictive variable
y_prime = 10*numpy.cos(X[:, 6])

# Fit an Earth model
model = Earth(max_degree=2, minspan_alpha=.5, smooth=True)
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

# Get the predicted values and derivatives
y_hat = model.predict(X)
y_prime_hat = model.predict_deriv(X, 'x6')

# Plot true and predicted function values and derivatives
# for the predictive variable
plt.subplot(211)
plt.plot(X[:, 6], y, 'r.')
plt.plot(X[:, 6], y_hat, 'b.')
plt.ylabel('function')
plt.subplot(212)
plt.plot(X[:, 6], y_prime, 'r.')
plt.plot(X[:, 6], y_prime_hat[:, 0], 'b.')
plt.ylabel('derivative')
plt.show()
