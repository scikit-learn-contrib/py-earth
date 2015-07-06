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
y2 = 10 * \
    numpy.abs(numpy.sin((X[:, 5]) / 6) - 1.0)

y1 = (y1 - y1.mean()) / y1.std()
y2 = (y2 - y2.mean()) / y2.std()
y_mix = numpy.concatenate((y1[:, numpy.newaxis], y2[:, numpy.newaxis]), axis=1)

alphas = [1., 0.8, 0.4, 0.]
n_plots = len(alphas)
k = 0
fig = plt.figure()
for alpha in alphas:
    # Fit an Earth model
    model = Earth(max_degree=10,
                  minspan_alpha=.05,
                  endspan_alpha=.05,
                  max_terms=8,
                  check_every=1,
                  thresh=0.)
    output_weight = numpy.array([alpha, 1 - alpha])
    model.fit(X, y_mix, output_weight=output_weight)

    # Plot the model
    y_hat = model.predict(X)

    mse = ((y_hat - y_mix) ** 2).mean(axis=0)
    plt.subplot(n_plots, 2, k)
    plt.plot(X[:, 6], y_mix[:, 0], 'r.')
    plt.plot(X[:, 6], model.predict(X)[:, 0], 'b.')
    plt.title("MSE: {0:.2f}, Weight : {1:.1f}".format(mse[0], alpha))
    plt.subplot(n_plots, 2, k + 1)
    plt.plot(X[:, 5], y_mix[:, 1], 'r.')
    plt.plot(X[:, 5], model.predict(X)[:, 1], 'b.')
    plt.title("MSE: {0:.2f}, Weight : {1:.1f}".format(mse[1], 1 - alpha))
    k += 2

plt.savefig("out.png")
