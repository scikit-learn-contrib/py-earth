from pyearth._basis import (Basis, ConstantBasisFunction, HingeBasisFunction,
                            LinearBasisFunction)
from pyearth.export import export_python_function, export_python_string,\
    export_sympy
from nose.tools import assert_almost_equal
import numpy
import six
from pyearth import Earth
from pyearth._types import BOOL
from pyearth.test.testing_utils import if_pandas,\
    if_sympy
from itertools import product
from numpy.testing.utils import assert_array_almost_equal

numpy.random.seed(0)

basis = Basis(10)
constant = ConstantBasisFunction()
basis.append(constant)
bf1 = HingeBasisFunction(constant, 0.1, 10, 1, False, 'x1')
bf2 = HingeBasisFunction(constant, 0.1, 10, 1, True, 'x1')
bf3 = LinearBasisFunction(bf1, 2, 'x2')
basis.append(bf1)
basis.append(bf2)
basis.append(bf3)
X = numpy.random.normal(size=(100, 10))
missing = numpy.zeros_like(X, dtype=BOOL)
B = numpy.empty(shape=(100, 4), dtype=numpy.float64)
basis.transform(X, missing, B)
beta = numpy.random.normal(size=4)
y = numpy.empty(shape=100, dtype=numpy.float64)
y[:] = numpy.dot(B, beta) + numpy.random.normal(size=100)
beta2 = numpy.random.normal(size=4)
y2 = numpy.empty(shape=100, dtype=numpy.float64)
y2[:] = numpy.dot(B, beta2) + numpy.random.normal(size=100)
Y = numpy.concatenate([y[:, None], y2[:, None]], axis=1)
default_params = {"penalty": 1}


def test_export_python_function():
    for smooth in (True, False):
        model = Earth(penalty=1, smooth=smooth, max_degree=2).fit(X, y)
        export_model = export_python_function(model)
        for exp_pred, model_pred in zip(model.predict(X), export_model(X)):
            assert_almost_equal(exp_pred, model_pred)


def test_export_python_string():
    for smooth in (True, False):
        model = Earth(penalty=1, smooth=smooth, max_degree=2).fit(X, y)
        export_model = export_python_string(model, 'my_test_model')
        six.exec_(export_model, globals())
        for exp_pred, model_pred in zip(model.predict(X), my_test_model(X)):
            assert_almost_equal(exp_pred, model_pred)

@if_pandas
@if_sympy
def test_export_sympy():
    import pandas as pd
    from sympy.utilities.lambdify import lambdify
    from sympy.printing.lambdarepr import NumPyPrinter

    class PyEarthNumpyPrinter(NumPyPrinter):
        def _print_Max(self, expr):
            return 'maximum(' + ','.join(self._print(i) for i in expr.args) + ')'

        def _print_NaNProtect(self, expr):
            return 'where(isnan(' + ','.join(self._print(a) for a in expr.args) + '), 0, ' \
                + ','.join(self._print(a) for a in expr.args) + ')'

        def _print_Missing(self, expr):
            return 'isnan(' + ','.join(self._print(a) for a in expr.args) + ').astype(float)'

    for smooth, n_cols, allow_missing in product((True, False), (1, 2), (True, False)):
        X_df = pd.DataFrame(X.copy(), columns=['x_%d' % i for i in range(X.shape[1])])
        y_df = pd.DataFrame(Y[:, :n_cols])
        if allow_missing:
            # Randomly remove some values so that the fitted model contains MissingnessBasisFunctions
            X_df['x_1'][numpy.random.binomial(n=1, p=.1, size=X_df.shape[0]).astype(bool)] = numpy.nan

        model = Earth(allow_missing=allow_missing, smooth=smooth, max_degree=2).fit(X_df, y_df)
        expressions = export_sympy(model) if n_cols > 1 else [export_sympy(model)]
        module_dict = {'select': numpy.select, 'less_equal': numpy.less_equal, 'isnan': numpy.isnan,
                       'greater_equal':numpy.greater_equal, 'logical_and': numpy.logical_and, 'less': numpy.less,
                       'logical_not':numpy.logical_not, "greater": numpy.greater, 'maximum':numpy.maximum,
                       'Missing': lambda x: numpy.isnan(x).astype(float),
                       'NaNProtect': lambda x: numpy.where(numpy.isnan(x), 0, x), 'nan': numpy.nan,
                       'float': float, 'where': numpy.where
                       }

        for i, expression in enumerate(expressions):
            # The lambdified functions for smoothed basis functions only work with modules='numpy' and
            # for regular basis functions with modules={'Max':numpy.maximum}.  This is a confusing situation
            func = lambdify(X_df.columns, expression, printer=PyEarthNumpyPrinter, modules=module_dict)
            y_pred_sympy = func(*[X_df.loc[:,var] for var in X_df.columns])

            y_pred = model.predict(X_df)[:,i] if n_cols > 1 else model.predict(X_df)
            assert_array_almost_equal(y_pred, y_pred_sympy)
