'''
Created on Feb 16, 2013

@author: jasonrudy
'''
from .earth import Earth

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
