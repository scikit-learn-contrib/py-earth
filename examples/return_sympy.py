"""
=========================================================================
Exporting a fitted Earth models as a sympy expression and generate C code
=========================================================================

A simple example in which an Earth model is fitted, exported as a sympy 
expression, and used to generate C code for prediction.  This example 
requires sympy.

"""

import numpy
from pyearth import Earth
from pyearth import export
from sympy.utilities.codegen import codegen

# Create some fake data
numpy.random.seed(2)
m = 1000
n = 10
X = 10 * numpy.random.uniform(size=(m, n)) - 40
y = 100 * \
    (numpy.sin((X[:, 6])) - 4.0) + \
    10 * numpy.random.normal(size=m)

# Fit an Earth model
model = Earth(max_degree=2, minspan_alpha=.5, verbose=False)
model.fit(X, y)

print(model.summary())

# Generate a sympy expression from the Earth object
print('Resulting sympy expression:')
expression = export.export_sympy(model)
print(expression)

# Generate C code from the sympy expression
(c_name, c_code), (h_name, h_code) = codegen(('model_predict', expression), 'C')
print(h_code)
print(c_code)