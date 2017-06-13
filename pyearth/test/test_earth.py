'''
Created on Feb 24, 2013

@author: jasonrudy
'''
import pickle
import copy
import os
from .testing_utils import (if_statsmodels, if_pandas, if_patsy,
                            if_environ_has, assert_list_almost_equal_value,
                            assert_list_almost_equal,
                            if_sklearn_version_greater_than_or_equal_to,
                            if_platform_not_win_32)
from nose.tools import (assert_equal, assert_true, assert_almost_equal,
                        assert_list_equal, assert_raises, assert_not_equal)
import numpy
from scipy.sparse import csr_matrix
from pyearth._types import BOOL
from pyearth._basis import (Basis, ConstantBasisFunction,
                            HingeBasisFunction, LinearBasisFunction)
from pyearth import Earth
import pyearth
from numpy.testing.utils import assert_array_almost_equal

regenerate_target_files = False

numpy.random.seed(1)
basis = Basis(10)
constant = ConstantBasisFunction()
basis.append(constant)
bf1 = HingeBasisFunction(constant, 0.1, 10, 1, False, 'x1')
bf2 = HingeBasisFunction(constant, 0.1, 10, 1, True, 'x1')
bf3 = LinearBasisFunction(bf1, 2, 'x2')
basis.append(bf1)
basis.append(bf2)
basis.append(bf3)
X = numpy.random.normal(size=(1000, 10))
missing = numpy.zeros_like(X, dtype=BOOL)
B = numpy.empty(shape=(1000, 4), dtype=numpy.float64)
basis.transform(X, missing, B)
beta = numpy.random.normal(size=4)
y = numpy.empty(shape=1000, dtype=numpy.float64)
y[:] = numpy.dot(B, beta) + numpy.random.normal(size=1000)
default_params = {"penalty": 1}

@if_platform_not_win_32
@if_sklearn_version_greater_than_or_equal_to('0.17.2')
def test_check_estimator():
    numpy.random.seed(0)
    import sklearn.utils.estimator_checks
    sklearn.utils.estimator_checks.MULTI_OUTPUT.append('Earth')
    sklearn.utils.estimator_checks.check_estimator(Earth)


def test_get_params():
    assert_equal(
        Earth().get_params(), {'penalty': None, 'min_search_points': None,
                               'endspan_alpha': None, 'check_every': None,
                               'max_terms': None, 'max_degree': None,
                               'minspan_alpha': None, 'thresh': None,
                               'zero_tol': None,
                               'minspan': None, 'endspan': None,
                               'allow_linear': None,
                               'use_fast': None, 'fast_K': None,
                               'fast_h': None, 'smooth': None,
                               'enable_pruning': True,
                               'allow_missing': False,
                               'feature_importance_type': None,
                               'verbose': False})
    assert_equal(
        Earth(
            max_degree=3).get_params(), {'penalty': None,
                                         'min_search_points': None,
                                         'endspan_alpha': None,
                                         'check_every': None,
                                         'max_terms': None, 'max_degree': 3,
                                         'minspan_alpha': None,
                                         'thresh': None, 'zero_tol': None,
                                         'minspan': None,
                                         'endspan': None,
                                         'allow_linear': None,
                                         'use_fast': None,
                                         'fast_K': None, 'fast_h': None,
                                         'smooth': None,
                                         'enable_pruning': True,
                                         'allow_missing': False,
                                         'feature_importance_type': None,
                                         'verbose': False})


@if_statsmodels
def test_linear_fit():
    from statsmodels.regression.linear_model import GLS, OLS

    earth = Earth(**default_params)
    earth.fit(X, y)
    earth.linear_fit(X, y)
    soln = OLS(y, earth.transform(X)).fit().params
    assert_almost_equal(numpy.mean((earth.coef_ - soln) ** 2), 0.0)

    sample_weight = 1.0 / (numpy.random.normal(size=y.shape) ** 2)
    earth.fit(X, y)
    earth.linear_fit(X, y, sample_weight)
    soln = GLS(y, earth.transform(
        X), 1.0 / sample_weight).fit().params
    assert_almost_equal(numpy.mean((earth.coef_ - soln) ** 2), 0.0)


def test_sample_weight():
    group = numpy.random.binomial(1, .5, size=1000) == 1
    sample_weight = 1 / (group * 100 + 1.0)
    x = numpy.random.uniform(-10, 10, size=1000)
    y = numpy.abs(x)
    y[group] = numpy.abs(x[group] - 5)
    y += numpy.random.normal(0, 1, size=1000)
    model = Earth().fit(x[:, numpy.newaxis], y, sample_weight=sample_weight)

    # Check that the model fits better for the more heavily weighted group
    assert_true(model.score(x[group], y[group]) < model.score(
        x[numpy.logical_not(group)], y[numpy.logical_not(group)]))

    # Make sure that the score function gives the same answer as the trace
    pruning_trace = model.pruning_trace()
    rsq_trace = pruning_trace.rsq(model.pruning_trace().get_selected())
    assert_almost_equal(model.score(x, y, sample_weight=sample_weight),
                        rsq_trace)

    # Uncomment below to see what this test situation looks like
#     from matplotlib import pyplot
#     print model.summary()
#     print model.score(x,y,sample_weight = sample_weight)
#     pyplot.figure()
#     pyplot.plot(x,y,'b.')
#     pyplot.plot(x,model.predict(x),'r.')
#     pyplot.show()


def test_output_weight():
    x = numpy.random.uniform(-1, 1, size=(1000, 1))
    y = (numpy.dot(x, numpy.random.normal(0, 1, size=(1, 10)))) ** 5 + 1
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    group = numpy.array([1] * 5 + [0] * 5)
    output_weight = numpy.array([1] * 5 + [2] * 5, dtype=float)
    model = Earth().fit(x, y, output_weight=output_weight)

    # Check that the model fits at least better
    # the more heavily weighted group
    mse = ((model.predict(x) - y)**2).mean(axis=0)
    group1_mean = mse[group].mean()
    group2_mean = mse[numpy.logical_not(group)].mean()
    assert_true(group1_mean > group2_mean or
                round(abs(group1_mean - group2_mean), 7) == 0)


def test_missing_data():
    numpy.random.seed(0)
    earth = Earth(allow_missing=True, **default_params)
    missing_ = numpy.random.binomial(1, .05, X.shape).astype(bool)
    X_ = X.copy()
    X_[missing_] = None
    earth.fit(X_, y)
    res = str(earth.score(X_, y))
    filename = os.path.join(os.path.dirname(__file__),
                            'earth_regress_missing_data.txt')
    if regenerate_target_files:
        with open(filename, 'w') as fl:
            fl.write(res)
    with open(filename, 'r') as fl:
        prev = fl.read()
    try:
        assert_true(abs(float(res) - float(prev)) < .03)
    except AssertionError:
        print('Got %f, %f' % (float(res), float(prev)))
        raise

def test_fit():
    numpy.random.seed(0)
    earth = Earth(**default_params)
    earth.fit(X, y)
    res = str(earth.rsq_)
    filename = os.path.join(os.path.dirname(__file__),
                            'earth_regress.txt')
    if regenerate_target_files:
        with open(filename, 'w') as fl:
            fl.write(res)
    with open(filename, 'r') as fl:
        prev = fl.read()
    assert_true(abs(float(res) - float(prev)) < .05)


def test_smooth():
    numpy.random.seed(0)
    model = Earth(penalty=1, smooth=True)
    model.fit(X, y)
    res = str(model.rsq_)
    filename = os.path.join(os.path.dirname(__file__),
                            'earth_regress_smooth.txt')
    if regenerate_target_files:
        with open(filename, 'w') as fl:
            fl.write(res)
    with open(filename, 'r') as fl:
        prev = fl.read()
    assert_true(abs(float(res) - float(prev)) < .05)


def test_linvars():
    earth = Earth(**default_params)
    earth.fit(X, y, linvars=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    res = str(earth.rsq_)
    filename = os.path.join(os.path.dirname(__file__),
                            'earth_linvars_regress.txt')
    if regenerate_target_files:
        with open(filename, 'w') as fl:
            fl.write(res)
    with open(filename, 'r') as fl:
        prev = fl.read()

    assert_equal(res, prev)


def test_linvars_coefs():
    nb_vars = 11
    coefs = numpy.random.uniform(size=(nb_vars,))
    X = numpy.random.uniform(size=(100, nb_vars))
    bias = 1
    y = numpy.dot(X, coefs[:, numpy.newaxis]) + bias
    earth = Earth(max_terms=nb_vars * 2,
                  max_degree=1,
                  enable_pruning=False,
                  check_every=1,
                  thresh=0,
                  minspan=1,
                  endspan=1).fit(X, y, linvars=range(nb_vars))
    earth_bias = earth.coef_[0, 0]
    earth_coefs = sorted(earth.coef_[1:])

    assert_almost_equal(earth_bias, bias)
    assert_list_almost_equal(earth_coefs, sorted(coefs))


def test_score():
    earth = Earth(**default_params)
    model = earth.fit(X, y)
    record = model.pruning_trace()
    rsq = record.rsq(record.get_selected())
    assert_almost_equal(rsq, model.score(X, y))


@if_pandas
@if_environ_has('test_pathological_cases')
def test_pathological_cases():
    import pandas
    directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'pathological_data')
    cases = {'issue_44': {},
             'issue_50': {'penalty': 0.5,
                          'minspan': 1,
                          'allow_linear': False,
                          'endspan': 1,
                          'check_every': 1,
                          'sample_weight': 'issue_50_weight.csv'}}
    for case, settings in cases.iteritems():
        data = pandas.read_csv(os.path.join(directory, case + '.csv'))
        y = data['y']
        del data['y']
        X = data
        if 'sample_weight' in settings:
            filename = os.path.join(directory, settings['sample_weight'])
            sample_weight = pandas.read_csv(filename)['sample_weight']
            del settings['sample_weight']
        else:
            sample_weight = None
        model = Earth(**settings)
        model.fit(X, y, sample_weight=sample_weight)
        with open(os.path.join(directory, case + '.txt'), 'r') as infile:
            correct = infile.read()
        assert_equal(model.summary(), correct)


@if_pandas
def test_pandas_compatibility():
    import pandas
    X_df = pandas.DataFrame(X)
    y_df = pandas.DataFrame(y)
    colnames = ['xx' + str(i) for i in range(X.shape[1])]
    X_df.columns = colnames

    earth = Earth(**default_params)
    model = earth.fit(X_df, y_df)
    assert_list_equal(
        colnames, model.forward_trace()._getstate()['xlabels'])


@if_patsy
@if_pandas
def test_patsy_compatibility():
    import pandas
    import patsy
    X_df = pandas.DataFrame(X)
    y_df = pandas.DataFrame(y)
    colnames = ['xx' + str(i) for i in range(X.shape[1])]
    X_df.columns = colnames
    X_df['y'] = y
    y_df, X_df = patsy.dmatrices(
        'y ~ xx0 + xx1 + xx2 + xx3 + xx4 + xx5 + xx6 + xx7 + xx8 + xx9 - 1',
        data=X_df)

    model = Earth(**default_params).fit(X_df, y_df)
    assert_list_equal(
        colnames, model.forward_trace()._getstate()['xlabels'])


def test_pickle_compatibility():
    earth = Earth(**default_params)
    model = earth.fit(X, y)
    model_copy = pickle.loads(pickle.dumps(model))
    assert_true(model_copy == model)
    assert_array_almost_equal(model.predict(X), model_copy.predict(X))
    assert_true(model.basis_[0] is model.basis_[1]._get_root())
    assert_true(model_copy.basis_[0] is model_copy.basis_[1]._get_root())


def test_pickle_version_storage():
    earth = Earth(**default_params)
    model = earth.fit(X, y)
    assert_equal(model._version, pyearth.__version__)
    model._version = 'hello'
    assert_equal(model._version,'hello')
    model_copy = pickle.loads(pickle.dumps(model))
    assert_equal(model_copy._version, model._version)


def test_copy_compatibility():
    numpy.random.seed(0)
    model = Earth(**default_params).fit(X, y)
    model_copy = copy.copy(model)
    assert_true(model_copy == model)
    assert_array_almost_equal(model.predict(X), model_copy.predict(X))
    assert_true(model.basis_[0] is model.basis_[1]._get_root())
    assert_true(model_copy.basis_[0] is model_copy.basis_[1]._get_root())


def test_exhaustive_search():
    model = Earth(max_terms=13,
                  enable_pruning=False,
                  check_every=1,
                  thresh=0,
                  minspan=1,
                  endspan=1)
    model.fit(X, y)
    assert_equal(model.basis_.plen(), model.coef_.shape[1])
    assert_equal(model.transform(X).shape[1], len(model.basis_))


def test_nb_terms():

    for max_terms in (1, 3, 12, 13):
        model = Earth(max_terms=max_terms)
        model.fit(X, y)
        assert_true(len(model.basis_) <= max_terms + 2)
        assert_true(len(model.coef_) <= len(model.basis_))
        assert_true(len(model.coef_) >= 1)
        if max_terms == 1:
            assert_list_almost_equal_value(model.predict(X), y.mean())


def test_nb_degrees():
    for max_degree in (1, 2, 12, 13):
        model = Earth(max_terms=10,
                      max_degree=max_degree,
                      enable_pruning=False,
                      check_every=1,
                      thresh=0,
                      minspan=1,
                      endspan=1)
        model.fit(X, y)
        for basis in model.basis_:
            assert_true(basis.degree() >= 0)
            assert_true(basis.degree() <= max_degree)


def test_eq():
    model1 = Earth(**default_params)
    model2 = Earth(**default_params)
    assert_equal(model1, model2)
    assert_not_equal(model1, 5)

    params = {}
    params.update(default_params)
    params["penalty"] = 15
    model2 = Earth(**params)
    assert_not_equal(model1, model2)

    model3 = Earth(**default_params)
    model3.unknown_parameter = 5
    assert_not_equal(model1, model3)


def test_sparse():
    X_sparse = csr_matrix(X)

    model = Earth(**default_params)
    assert_raises(TypeError, model.fit, X_sparse, y)

    model = Earth(**default_params)
    model.fit(X, y)
    assert_raises(TypeError, model.predict, X_sparse)
    assert_raises(TypeError, model.predict_deriv, X_sparse)
    assert_raises(TypeError, model.transform, X_sparse)
    assert_raises(TypeError, model.score, X_sparse)

    model = Earth(**default_params)
    sample_weight = csr_matrix([1.] * X.shape[0])
    assert_raises(TypeError, model.fit, X, y, sample_weight)


def test_shape():
    model = Earth(**default_params)
    model.fit(X, y)

    X_reduced = X[:, 0:5]
    assert_raises(ValueError, model.predict, X_reduced)
    assert_raises(ValueError, model.predict_deriv, X_reduced)
    assert_raises(ValueError, model.transform, X_reduced)
    assert_raises(ValueError, model.score, X_reduced)

    model = Earth(**default_params)
    X_subsampled = X[0:10]
    assert_raises(ValueError, model.fit, X_subsampled, y)

    model = Earth(**default_params)
    y_subsampled = X[0:10]
    assert_raises(ValueError, model.fit, X, y_subsampled)

    model = Earth(**default_params)
    sample_weights = numpy.array([1.] * len(X))
    sample_weights_subsampled = sample_weights[0:10]
    assert_raises(ValueError, model.fit, X, y, sample_weights_subsampled)


def test_deriv():

    model = Earth(**default_params)
    model.fit(X, y)
    assert_equal(X.shape + (1,), model.predict_deriv(X).shape)
    assert_equal((X.shape[0], 1, 1), model.predict_deriv(X, variables=0).shape)
    assert_equal((X.shape[0], 1, 1), model.predict_deriv(
        X, variables='x0').shape)
    assert_equal((X.shape[0], 3, 1),
                 model.predict_deriv(X, variables=[1, 5, 7]).shape)
    assert_equal((X.shape[0], 0, 1),
                 model.predict_deriv(X, variables=[]).shape)

    res_deriv = model.predict_deriv(X, variables=['x2', 'x7', 'x0', 'x1'])
    assert_equal((X.shape[0], 4, 1), res_deriv.shape)

    res_deriv = model.predict_deriv(X, variables=['x0'])
    assert_equal((X.shape[0], 1, 1), res_deriv.shape)

    assert_equal((X.shape[0], 1, 1),
                 model.predict_deriv(X, variables=[0]).shape)


def test_xlabels():

    model = Earth(**default_params)
    assert_raises(ValueError, model.fit, X[
                  :, 0:5], y, xlabels=['var1', 'var2'])

    model = Earth(**default_params)
    model.fit(X[:, 0:3], y, xlabels=['var1', 'var2', 'var3'])

    model = Earth(**default_params)
    model.fit(X[:, 0:3], y, xlabels=['var1', 'var2', 'var3'])


def test_untrained():
    # NotFittedError moved from utils.validation to exceptions
    # some time after 0.17.1
    try:
        from sklearn.exceptions import NotFittedError
    except ImportError:
        from sklearn.utils.validation import NotFittedError

    # Make sure calling methods that require a fitted Earth object
    # raises the appropriate exception when using a not yet fitted
    # Earth object
    model = Earth(**default_params)
    assert_raises(NotFittedError, model.predict, X)
    assert_raises(NotFittedError, model.transform, X)
    assert_raises(NotFittedError, model.predict_deriv, X)
    assert_raises(NotFittedError, model.score, X)

    # the following should be changed to raise NotFittedError
    assert_equal(model.forward_trace(), None)
    assert_equal(model.pruning_trace(), None)
    assert_equal(model.summary(), "Untrained Earth Model")


def test_fast():
    earth = Earth(max_terms=10,
                  max_degree=5,
                  **default_params)
    earth.fit(X, y)
    normal_summary = earth.summary()
    earth = Earth(use_fast=True,
                  max_terms=10,
                  max_degree=5,
                  fast_K=10,
                  fast_h=1,
                  **default_params)
    earth.fit(X, y)
    fast_summary = earth.summary()
    assert_equal(normal_summary, fast_summary)


def test_feature_importance():
    criteria = ('rss', 'gcv', 'nb_subsets')
    for imp in criteria:
        earth = Earth(feature_importance_type=imp, **default_params)
        earth.fit(X, y)
        assert len(earth.feature_importances_) == X.shape[1]
    earth = Earth(feature_importance_type=criteria, **default_params)
    earth.fit(X, y)
    assert type(earth.feature_importances_) == dict
    assert set(earth.feature_importances_.keys()) == set(criteria)
    for crit, val in earth .feature_importances_.items():
        assert len(val) == X.shape[1]

    assert_raises(
            ValueError,
            Earth(feature_importance_type='bad_name', **default_params).fit,
            X, y)

    earth = Earth(feature_importance_type=('rss',), **default_params)
    earth.fit(X, y)
    assert len(earth.feature_importances_) == X.shape[1]

    assert_raises(
            ValueError,
            Earth(feature_importance_type='rss', enable_pruning=False, **default_params).fit,
            X, y)
