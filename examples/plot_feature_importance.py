"""
===========================
Plotting feature importance
===========================

A simple example showing how to compute and display
feature importances, it is also compared with the
feature importances obtained using random forests.

Feature importance is a measure of the effect of the features
on the outputs. For each feature, the values go from
0 to 1 where a higher the value means that the feature will have
a higher effect on the outputs.

Currently three criteria are supported : 'gcv', 'rss' and 'nb_subsets'.
See [1], section 12.3 for more information about the criteria.

.. [1] http://www.milbo.org/doc/earth-notes.pdf

"""
import numpy
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from pyearth import Earth

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10

X = numpy.random.uniform(size=(m, n))
y = (10 * numpy.sin(numpy.pi * X[:, 0] * X[:, 1]) +
     20 * (X[:, 2] - 0.5) ** 2 +
     10 * X[:, 3] +
     5 * X[:, 4] + numpy.random.uniform(size=m))
# Fit an Earth model
criteria = ('rss', 'gcv', 'nb_subsets')
model = Earth(max_degree=3,
              max_terms=10,
              minspan_alpha=.5,
              feature_importance_type=criteria,
              verbose=True)
model.fit(X, y)
rf = RandomForestRegressor()
rf.fit(X, y)
# Print the model
print(model.trace())
print(model.summary())
print(model.summary_feature_importances(sort_by='gcv'))

# Plot the feature importances
importances = model.feature_importances_
importances['random_forest'] = rf.feature_importances_
criteria = criteria + ('random_forest',)
idx = 1

fig = plt.figure(figsize=(20, 10))
labels = ['$x_{}$'.format(i) for i in range(n)]
for crit in criteria:
    plt.subplot(2, 2, idx)
    plt.bar(numpy.arange(len(labels)),
            importances[crit],
            align='center',
            color='red')
    plt.xticks(numpy.arange(len(labels)), labels)
    plt.title(crit)
    plt.ylabel('importances')
    idx += 1
title = '$x_0,...x_9 \sim \mathcal{N}(0, 1)$\n$y= 10sin(\pi x_{0}x_{1}) + 20(x_2 - 0.5)^2 + 10x_3 + 5x_4 + Unif(0, 1)$'
fig.suptitle(title, fontsize="x-large")
plt.show()
