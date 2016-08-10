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
    if_sympy, assert_list_almost_equal

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
    
    for smooth in (True, False):
        X_df = pd.DataFrame(X, columns=['x_%d' % i for i in range(X.shape[1])])
        model = Earth(penalty=1, smooth=smooth, max_degree=2, max_terms=80).fit(X_df, y)
        expression = export_sympy(model)
        
        # The lambdified functions for smoothed basis functions only work with modules='numpy' and 
        # for regular basis functions with modules={'Max':numpy.maximum}.  This is a confusing situation 
        func = lambdify(X_df.columns, expression, modules=['numpy' if smooth else {'Max':numpy.maximum}])
        y_pred_sympy = func(*[X_df.loc[:,var] for var in X_df.columns])
                
        y_pred = model.predict(X_df)
        assert_list_almost_equal(y_pred, y_pred_sympy)
