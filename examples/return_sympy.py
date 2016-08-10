"""
============================================
Returning a sympy expression for a  simple sine function
============================================

A simple example returning a sympy expression describing the fit of a sine function computed by Earth.

"""

import numpy
import matplotlib.pyplot as plt

from pyearth import Earth
from pyearth import export

# Create some fake data
numpy.random.seed(2)
m = 1000
n = 10
X = 80 * numpy.random.uniform(size=(m, n)) - 40
y = 100 * \
    (numpy.sin((X[:, 6])) - 4.0) + \
    10 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth(max_degree=2, minspan_alpha=.5, verbose=False)
model.fit(X, y)

print(model.summary())

#return sympy expression 
print(export.export_sympy(model))

