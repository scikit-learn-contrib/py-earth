from distutils.core import setup
from distutils.extension import Extension
import numpy
import sys
import os
sys.path.insert(0,os.path.join('.','pyearth'))
from _version import __version__

#Determine whether to use Cython
if '--cythonize' in sys.argv:
    cythonize_switch = True
    del sys.argv[sys.argv.index('--cythonize')]
else:
    cythonize_switch = False

#Find all includes
local_inc = 'pyearth'
numpy_inc = numpy.get_include()

#Set up the ext_modules for Cython or not, depending
if cythonize_switch:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension("pyearth._util", ["pyearth/_util.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._basis", ["pyearth/_basis.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._record", ["pyearth/_record.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._pruning", ["pyearth/_pruning.pyx"],include_dirs = [local_inc, numpy_inc]),
                             Extension("pyearth._forward", ["pyearth/_forward.pyx"],include_dirs = [local_inc, numpy_inc])
                             ])
else:
    ext_modules = [Extension("pyearth._util", ["pyearth/_util.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._basis", ["pyearth/_basis.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._record", ["pyearth/_record.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._pruning", ["pyearth/_pruning.c"],include_dirs = [local_inc, numpy_inc]),
                   Extension("pyearth._forward", ["pyearth/_forward.c"],include_dirs = [local_inc, numpy_inc])
                   ]
    
#Create a dictionary of arguments for setup
setup_args = {'name':'py-earth',
    'version':__version__,
    'author':'Jason Rudy',
    'author_email':'jcrudy@gmail.com',
    'packages':['pyearth','pyearth.test'],
    'license':'LICENSE.txt',
    'description':'A Python implementation of Jerome Friedman\'s MARS algorithm.',
    'long_description':open('README.md','r').read(),
    'py_modules' : ['pyearth.earth','pyearth._version'],
    'ext_modules' : ext_modules,
    'scripts':[os.path.join('scripts','pyearth_vs_earth.py')],
    'classifiers' : ['Development Status :: 3 - Alpha'],
    'requires':['numpy','sklearn'],
    'tests_require':['nose']} 

#Add the build_ext command only if cythonizing
if cythonize_switch:
    setup_args['cmdclass'] = {'build_ext': build_ext}

#Finally
setup(**setup_args)
