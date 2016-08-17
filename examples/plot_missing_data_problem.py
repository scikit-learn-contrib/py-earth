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
missing = numpy.random.binomial(1, .2, (m, n)).astype(bool)
X_full = X.copy()
X[missing] = None
idx5 = (1 - missing[:, 5]).astype(bool)
idx6 = (1 - missing[:, 6]).astype(bool)

# Fit an Earth model
model = Earth(max_degree=5, minspan_alpha=.5, allow_missing=True,
              enable_pruning=True, thresh=.001, smooth=True, verbose=2)
model.fit(X, y)
# Print the model
print(model.summary())

# Plot the model
y_hat = model.predict(X)
fig = plt.figure()

ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(X_full[idx5, 5], y[idx5], 'b.')
ax1.plot(X_full[idx5, 5], y_hat[idx5], 'r.')
ax1.set_xlim(-40, 40)
ax1.set_title('x5 present')
ax1.set_xlabel('x5')

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(X_full[idx6, 6], y[idx6], 'b.')
ax2.plot(X_full[idx6, 6], y_hat[idx6], 'r.')
ax2.set_xlim(-40, 40)
ax2.set_title('x6 present')
ax2.set_xlabel('x6')

ax3 = fig.add_subplot(3, 2, 3, sharex=ax1)
ax3.plot(X_full[~idx6, 5], y[~idx6], 'b.')
ax3.plot(X_full[~idx6, 5], y_hat[~idx6], 'r.')
ax3.set_title('x6 missing')
ax3.set_xlabel('x5')

ax4 = fig.add_subplot(3, 2, 4, sharex=ax2)
ax4.plot(X_full[~idx5, 6], y[~idx5], 'b.')
ax4.plot(X_full[~idx5, 6], y_hat[~idx5], 'r.')
ax4.set_title('x5 missing')
ax4.set_xlabel('x6')

ax5 = fig.add_subplot(3, 2, 5, sharex=ax1)
ax5.plot(X_full[(~idx6) & (~idx5), 5], y[(~idx6) & (~idx5)], 'b.')
ax5.plot(X_full[(~idx6) & (~idx5), 5], y_hat[(~idx6) & (~idx5)], 'r.')
ax5.set_title('both missing')
ax5.set_xlabel('x5')

ax6 = fig.add_subplot(3, 2, 6, sharex=ax2)
ax6.plot(X_full[(~idx6) & (~idx5), 6], y[(~idx6) & (~idx5)], 'b.')
ax6.plot(X_full[(~idx6) & (~idx5), 6], y_hat[(~idx6) & (~idx5)], 'r.')
ax6.set_title('both missing')
ax6.set_xlabel('x6')

fig.tight_layout()
plt.show()
