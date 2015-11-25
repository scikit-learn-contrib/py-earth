"""
================================================================
Plotting sine function with redundant predictors an missing data
================================================================

An example plotting a fit of the sine function.  There are two 
redundant predictors, each of which has independent and random
missingness.
"""
import numpy
import matplotlib.pyplot as plt

from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
X[:, 5] = X[:, 6] + numpy.random.normal(0, .1, m)
y = 100 * \
    (numpy.sin((X[:, 5] + X[:, 6]) / 20) - 4.0) + \
    10 * numpy.random.normal(size=m)
missing = numpy.random.binomial(1, .3, (m, n)).astype(bool)
X[missing] = None
idx5 = (1 - missing[:, 5]).astype(bool)
idx6 = (1 - missing[:, 6]).astype(bool)

# Fit an Earth model
model = Earth(max_degree=3, minspan_alpha=.5, allow_missing=True, 
              enable_pruning=False)
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

# Plot the model
y_hat = model.predict(X)
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
ax.plot(X[idx5, 5], y[idx5], 'b.')
ax.plot(X[idx5, 5], y_hat[idx5], 'r.')


ax = fig.add_subplot(1, 2, 2)
ax.plot(X[idx6, 6], y[idx6], 'b.')
ax.plot(X[idx6, 6], y_hat[idx6], 'r.')

plt.show()
