'''
Created on Feb 24, 2013

@author: jasonrudy
'''
import numpy
from pyearth._basis import Basis, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
from pyearth import Earth, export_python_model, export_python_string, export_labelled_coefficients
import pickle
import copy
import os
from testing_utils import if_statsmodels, if_pandas, if_patsy, if_environ_has
from nose.tools import assert_equal, assert_not_equal, assert_true, assert_false, \
    assert_almost_equal, assert_list_equal


class TestEarth(object):

    def __init__(self):
        numpy.random.seed(0)
        self.basis = Basis(10)
        constant = ConstantBasisFunction()
        self.basis.append(constant)
        bf1 = HingeBasisFunction(constant, 0.1, 10, 1, False, 'x1')
        bf2 = HingeBasisFunction(constant, 0.1, 10, 1, True, 'x1')
        bf3 = LinearBasisFunction(bf1, 2, 'x2')
        self.basis.append(bf1)
        self.basis.append(bf2)
        self.basis.append(bf3)
        self.X = numpy.random.normal(size=(100, 10))
        self.B = numpy.empty(shape=(100, 4), dtype=numpy.float64)
        self.basis.transform(self.X, self.B)
        self.beta = numpy.random.normal(size=4)
        self.y = numpy.empty(shape=100, dtype=numpy.float64)
        self.y[:] = numpy.dot(
            self.B, self.beta) + numpy.random.normal(size=100)
        self.earth = Earth(penalty=1)

    def test_get_params(self):
        assert_equal(
            Earth().get_params(), {'penalty': None, 'min_search_points': None,
                                   'endspan_alpha': None, 'check_every': None,
                                   'max_terms': None, 'max_degree':
                                   None, 'minspan_alpha': None,
                                   'thresh': None, 'minspan': None, 'endspan': None,
                                   'allow_linear': None, 'smooth': None})
        assert_equal(
            Earth(
                max_degree=3).get_params(), {'penalty': None, 'min_search_points': None,
                                             'endspan_alpha': None,
                                             'check_every': None,
                                             'max_terms': None, 'max_degree':
                                             3, 'minspan_alpha': None,
                                             'thresh': None, 'minspan': None, 'endspan': None,
                                             'allow_linear': None, 'smooth': None})

    @if_statsmodels
    def test_linear_fit(self):
        from statsmodels.regression.linear_model import GLS, OLS
        self.earth.fit(self.X, self.y)
        self.earth._Earth__linear_fit(self.X, self.y)
        soln = OLS(self.y, self.earth.transform(self.X)).fit().params
        assert_almost_equal(numpy.mean((self.earth.coef_ - soln) ** 2), 0.0)

        sample_weight = 1.0 / (numpy.random.normal(size=self.y.shape) ** 2)
        self.earth.fit(self.X, self.y)
        self.earth._Earth__linear_fit(self.X, self.y, sample_weight)
        soln = GLS(self.y, self.earth.transform(
            self.X), 1.0 / sample_weight).fit().params
        assert_almost_equal(numpy.mean((self.earth.coef_ - soln) ** 2), 0.0)

    def test_sample_weight(self):
        group = numpy.random.binomial(1, .5, size=1000) == 1
        sample_weight = 1 / (group * 100 + 1.0)
        x = numpy.random.uniform(-10, 10, size=1000)
        y = numpy.abs(x)
        y[group] = numpy.abs(x[group] - 5)
        y += numpy.random.normal(0, 1, size=1000)
        model = Earth().fit(x, y, sample_weight=sample_weight)

        # Check that the model fits better for the more heavily weighted group
        assert_true(model.score(x[group], y[group]) < model.score(
            x[numpy.logical_not(group)], y[numpy.logical_not(group)]))

        # Make sure that the score function gives the same answer as the trace
        assert_almost_equal(model.score(x, y, sample_weight=sample_weight),
                            model.pruning_trace().rsq(model.pruning_trace().get_selected()))

        # Uncomment below to see what this test situation looks like
#        from matplotlib import pyplot
#        print model.summary()
#        print model.score(x,y,sample_weight = sample_weight)
#        pyplot.figure()
#        pyplot.plot(x,y,'b.')
#        pyplot.plot(x,model.predict(x),'r.')
#        pyplot.show()

    def test_fit(self):
        self.earth.fit(self.X, self.y)
        res = str(self.earth.trace()) + '\n' + self.earth.summary()
#        with open(os.path.join(os.path.dirname(__file__), 'earth_regress.txt'),'w') as fl:
#            fl.write(res)
        with open(os.path.join(os.path.dirname(__file__), 'earth_regress.txt'), 'r') as fl:
            prev = fl.read()
        assert_equal(res, prev)

    def test_smooth(self):
        model = Earth(penalty=1, smooth=True)
        model.fit(self.X, self.y)
        res = str(model.trace()) + '\n' + model.summary()
#         with open(os.path.join(os.path.dirname(__file__), 'earth_regress_smooth.txt'),'w') as fl:
#             fl.write(res)
        with open(os.path.join(os.path.dirname(__file__), 'earth_regress_smooth.txt'), 'r') as fl:
            prev = fl.read()
        assert_equal(res, prev)

    def test_linvars(self):
        self.earth.fit(self.X, self.y, linvars=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        res = str(self.earth.trace()) + '\n' + self.earth.summary()
#        with open(os.path.dirname(__file__), 'earth_linvars_regress.txt'),'w') as fl:
#            fl.write(res)
        with open(os.path.join(os.path.dirname(__file__), 'earth_linvars_regress.txt'), 'r') as fl:
            prev = fl.read()
        assert_equal(res, prev)

    def test_score(self):
        model = self.earth.fit(self.X, self.y)
        record = model.pruning_trace()
        rsq = record.rsq(record.get_selected())
        assert_almost_equal(rsq, model.score(self.X, self.y))

    @if_pandas
    @if_environ_has('test_pathological_cases')
    def test_pathological_cases(self):
        import pandas
        directory = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'pathological_data')
        cases = {'issue_44': {},
                 'issue_50': {'penalty': 0.5, 'minspan': 1, 'allow_linear': False,
                              'endspan': 1, 'check_every': 1, 'sample_weight': 'issue_50_weight.csv'}}
        for case, settings in cases.iteritems():
            data = pandas.read_csv(os.path.join(directory, case + '.csv'))
            y = data['y']
            del data['y']
            X = data
            if 'sample_weight' in settings:
                sample_weight = pandas.read_csv(os.path.join(directory, settings['sample_weight']))['sample_weight']
                del settings['sample_weight']
            else:
                sample_weight = None
            model = Earth(**settings).fit(X, y, sample_weight = sample_weight)
#             with open(os.path.join(directory, case + '.txt'), 'w') as outfile:
#                 outfile.write(model.summary())
            with open(os.path.join(directory, case + '.txt'), 'r') as infile:
                correct = infile.read()
            assert_equal(model.summary(), correct)

    @if_pandas
    def test_pandas_compatibility(self):
        import pandas
        X = pandas.DataFrame(self.X)
        y = pandas.DataFrame(self.y)
        colnames = ['xx' + str(i) for i in range(X.shape[1])]
        X.columns = colnames
        model = self.earth.fit(X, y)
        assert_list_equal(
            colnames, model.forward_trace()._getstate()['xlabels'])

    @if_patsy
    @if_pandas
    def test_patsy_compatibility(self):
        import pandas
        import patsy
        X = pandas.DataFrame(self.X)
        y = pandas.DataFrame(self.y)
        colnames = ['xx' + str(i) for i in range(X.shape[1])]
        X.columns = colnames
        X['y'] = y
        y, X = patsy.dmatrices(
            'y ~ xx0 + xx1 + xx2 + xx3 + xx4 + xx5 + xx6 + xx7 + xx8 + xx9 - 1', data=X)
        model = self.earth.fit(X, y)
        assert_list_equal(
            colnames, model.forward_trace()._getstate()['xlabels'])

    def test_pickle_compatibility(self):
        model = self.earth.fit(self.X, self.y)
        model_copy = pickle.loads(pickle.dumps(model))
        assert_true(model_copy == model)
        assert_true(
            numpy.all(model.predict(self.X) == model_copy.predict(self.X)))
        assert_true(model.basis_[0] is model.basis_[1]._get_root())
        assert_true(model_copy.basis_[0] is model_copy.basis_[1]._get_root())

    def test_copy_compatibility(self):
        model = self.earth.fit(self.X, self.y)
        model_copy = copy.copy(model)
        assert_true(model_copy == model)
        assert_true(
            numpy.all(model.predict(self.X) == model_copy.predict(self.X)))
        assert_true(model.basis_[0] is model.basis_[1]._get_root())
        assert_true(model_copy.basis_[0] is model_copy.basis_[1]._get_root())

    def test_export_model(self):
        model = self.earth.fit(self.X, self.y)
        export_model = export_python_model(self.earth)
        for exp_pred, model_pred in zip(model.predict(self.X), export_model(self.X)):
            assert_almost_equal(exp_pred, model_pred)

    def test_export_model_string(self):
        model = self.earth.fit(self.X, self.y)
        export_model = export_python_string(self.earth, 'my_test_model')
        exec export_model
        for exp_pred, model_pred in zip(model.predict(self.X), my_test_model(self.X)):
            assert_almost_equal(exp_pred, model_pred)


if __name__ == '__main__':
    import nose
    nose.run(argv=[__file__, '-s', '-v'])
