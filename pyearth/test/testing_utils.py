from functools import wraps
from nose import SkipTest
import os

def if_environ_has(var_name):
    """Test decorator that skips test if environment variable is not defined."""
    
    def if_environ(func):
        @wraps(func)
        def run_test(*args, **kwargs):
            if var_name in os.environ:
                return func(*args, **kwargs)
            else:
                raise SkipTest('Only run if %s environment variable is defined.' \
                               % var_name)
        return run_test
    return if_environ

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
