import os
from functools import wraps
from nose import SkipTest


def if_environ_has(var_name):
    """Test decorator that skips test if environment variable is not defined."""

    def if_environ(func):
        @wraps(func)
        def run_test(*args, **kwargs):
            if var_name in os.environ:
                return func(*args, **kwargs)
            else:
                raise SkipTest('Only run if %s environment variable is defined.'
                               % var_name)
        return run_test
    return if_environ

from nose.tools import assert_almost_equal


def if_statsmodels(func):
    """Test decorator that skips test if statsmodels not installed. """

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import statsmodels
        except ImportError:
            raise SkipTest('statsmodels not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def if_pandas(func):
    """Test decorator that skips test if pandas not installed. """

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import pandas
        except ImportError:
            raise SkipTest('pandas not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def if_patsy(func):
    """Test decorator that skips test if patsy not installed. """

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import patsy
        except ImportError:
            raise SkipTest('patsy not available.')
        else:
            return func(*args, **kwargs)
    return run_test


def assert_list_almost_equal(list1, list2):
    for el1, el2 in zip(list1, list2):
        assert_almost_equal(el1, el2)

def assert_list_almost_equal_value(list, value):
    for el in list:
        assert_almost_equal(el, value)
