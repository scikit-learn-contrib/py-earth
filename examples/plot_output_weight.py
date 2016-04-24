"""
=================================================================
Demonstrating a use of weights in outputs with two sine functions
=================================================================

Each row in the grid is a run of an earth model.
Each column is an output.
In each run, different weights are given to
the outputs.

"""
import numpy as np
import matplotlib.pyplot as plt
from pyearth import Earth

# Create some fake data
np.random.seed(2)
m = 10000
n = 10
X = 80 * np.random.uniform(size=(m, n)) - 40
y1 = 120 * np.abs(np.sin((X[:, 6]) / 6) - 1.0) + 15 * np.random.normal(size=m)
y2 = 120 * np.abs(np.sin((X[:, 5]) / 6) - 1.0) + 15 * np.random.normal(size=m)

y1 = (y1 - y1.mean()) / y1.std()
y2 = (y2 - y2.mean()) / y2.std()
y_mix = np.concatenate((y1[:, np.newaxis], y2[:, np.newaxis]), axis=1)

alphas = [0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
n_plots = len(alphas)
k = 1
fig = plt.figure(figsize=(10, 15))
for i, alpha in enumerate(alphas):
    # Fit an Earth model
    model = Earth(max_degree=5,
                  minspan_alpha=.05,
                  endspan_alpha=.05,
                  max_terms=10,
                  check_every=1,
                  thresh=0.)
    output_weight = np.array([alpha, 1 - alpha])
    model.fit(X, y_mix, output_weight=output_weight)
    print(model.summary())

    # Plot the model
    y_hat = model.predict(X)

    mse = ((y_hat - y_mix) ** 2).mean(axis=0)
    ax = plt.subplot(n_plots, 2, k)
    ax.set_ylabel("Run {0}".format(i + 1), rotation=0, labelpad=20)
    plt.plot(X[:, 6], y_mix[:, 0], 'r.')
    plt.plot(X[:, 6], model.predict(X)[:, 0], 'b.')
    plt.title("MSE: {0:.3f}, Weight : {1:.1f}".format(mse[0], alpha))
    plt.subplot(n_plots, 2, k + 1)
    plt.plot(X[:, 5], y_mix[:, 1], 'r.')
    plt.plot(X[:, 5], model.predict(X)[:, 1], 'b.')
    plt.title("MSE: {0:.3f}, Weight : {1:.1f}".format(mse[1], 1 - alpha))
    k += 2
plt.tight_layout()
plt.show()
