'''
===================================================================
Plotting a multicolumn regression problem that includes missingness
===================================================================

An example plotting a simultaneous fit of the sine and cosine functions.
There are two redundant predictors, each of which has independent and random
missingness.
'''

import numpy
import matplotlib.pyplot as plt

from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
X[:, 5] = X[:, 6] + numpy.random.normal(0, .1, m)
y1 = 100 * \
    (numpy.sin((X[:, 5] + X[:, 6]) / 20) - 4.0) + \
    10 * numpy.random.normal(size=m)
y2 = 100 * \
    (numpy.cos((X[:, 5] + X[:, 6]) / 20) - 4.0) + \
    10 * numpy.random.normal(size=m)
y = numpy.concatenate([y1[:, None], y2[:, None]], axis=1)
missing = numpy.random.binomial(1, .2, (m, n)).astype(bool)
X_full = X.copy()
X[missing] = None
idx5 = (1 - missing[:, 5]).astype(bool)
idx6 = (1 - missing[:, 6]).astype(bool)

# Fit an Earth model
model = Earth(max_degree=5, minspan_alpha=.5, allow_missing=True,
              enable_pruning=True, thresh=.001, smooth=True,
              verbose=True)
model.fit(X, y)

# Print the model
print(model.trace())
print(model.summary())

# Plot the model
y_hat = model.predict(X)
fig = plt.figure()

for j in [0, 1]:
    ax1 = fig.add_subplot(3, 4, 1 + 2*j)
    ax1.plot(X_full[idx5, 5], y[idx5, j], 'b.')
    ax1.plot(X_full[idx5, 5], y_hat[idx5, j], 'r.')
    ax1.set_xlim(-40, 40)
    ax1.set_title('x5 present')
    ax1.set_xlabel('x5')
    ax1.set_ylabel('sin' if j == 0 else 'cos')

    ax2 = fig.add_subplot(3, 4, 2 + 2*j)
    ax2.plot(X_full[idx6, 6], y[idx6, j], 'b.')
    ax2.plot(X_full[idx6, 6], y_hat[idx6, j], 'r.')
    ax2.set_xlim(-40, 40)
    ax2.set_title('x6 present')
    ax2.set_xlabel('x6')
    ax2.set_ylabel('sin' if j == 0 else 'cos')

    ax3 = fig.add_subplot(3, 4, 5 + 2*j, sharex=ax1)
    ax3.plot(X_full[~idx6, 5], y[~idx6, j], 'b.')
    ax3.plot(X_full[~idx6, 5], y_hat[~idx6, j], 'r.')
    ax3.set_title('x6 missing')
    ax3.set_xlabel('x5')
    ax3.set_ylabel('sin' if j == 0 else 'cos')

    ax4 = fig.add_subplot(3, 4, 6 + 2*j, sharex=ax2)
    ax4.plot(X_full[~idx5, 6], y[~idx5, j], 'b.')
    ax4.plot(X_full[~idx5, 6], y_hat[~idx5, j], 'r.')
    ax4.set_title('x5 missing')
    ax4.set_xlabel('x6')
    ax4.set_ylabel('sin' if j == 0 else 'cos')

    ax5 = fig.add_subplot(3, 4, 9 + 2*j, sharex=ax1)
    ax5.plot(X_full[(~idx6) & (~idx5), 5], y[(~idx6) & (~idx5), j], 'b.')
    ax5.plot(X_full[(~idx6) & (~idx5), 5], y_hat[(~idx6) & (~idx5), j], 'r.')
    ax5.set_title('both missing')
    ax5.set_xlabel('x5')
    ax5.set_ylabel('sin' if j == 0 else 'cos')

    ax6 = fig.add_subplot(3, 4, 10 + 2*j, sharex=ax2)
    ax6.plot(X_full[(~idx6) & (~idx5), 6], y[(~idx6) & (~idx5), j], 'b.')
    ax6.plot(X_full[(~idx6) & (~idx5), 6], y_hat[(~idx6) & (~idx5), j], 'r.')
    ax6.set_title('both missing')
    ax6.set_xlabel('x6')
    ax6.set_ylabel('sin' if j == 0 else 'cos')

fig.tight_layout()
plt.show()
