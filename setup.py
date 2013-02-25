from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from Cython.Build import cythonize
from numpy.distutils.system_info import get_info
import sys

#Determine whether to use Cython
if '--cythonize' in sys.argv:
    cythonize_switch = True
    del sys.argv[sys.argv.index('--cythonize')]
else:
    cythonize_switch = False

#Find all includes
local_inc = 'pyearth'
numpy_inc = numpy.get_include()
blas_inc = get_info('blas_opt')['extra_compile_args'][1][2:] #TODO: Is there a better way?
lapack_inc = blas_inc #TODO: Is this generally true?

#Set up the ext_modules for Cython or not, depending
if cythonize_switch:
    ext_modules = cythonize([Extension("pyearth._blas", ["pyearth/_blas.pyx"],include_dirs = [blas_inc]),
                             Extension("pyearth._lapack", ["pyearth/_lapack.pyx"],include_dirs = [lapack_inc]),
                             Extension("pyearth._choldate", ["pyearth/_choldate.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._util", ["pyearth/_util.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._basis", ["pyearth/_basis.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._record", ["pyearth/_record.pyx"],include_dirs = [numpy_inc]),
                             Extension("pyearth._pruning", ["pyearth/_pruning.pyx"],include_dirs = [local_inc, numpy_inc]),
                             Extension("pyearth._forward", ["pyearth/_forward.pyx"],include_dirs = [local_inc, numpy_inc, blas_inc])
                             ])
else:
    ext_modules = [Extension("pyearth._blas", ["pyearth/_blas.c"],include_dirs = [blas_inc]),
                   Extension("pyearth._lapack", ["pyearth/_lapack.c"],include_dirs = [lapack_inc]),
                   Extension("pyearth._choldate", ["pyearth/_choldate.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._util", ["pyearth/_util.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._basis", ["pyearth/_basis.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._record", ["pyearth/_record.c"],include_dirs = [numpy_inc]),
                   Extension("pyearth._pruning", ["pyearth/_pruning.c"],include_dirs = [local_inc, numpy_inc]),
                   Extension("pyearth._forward", ["pyearth/_forward.c"],include_dirs = [local_inc, numpy_inc, blas_inc])
                   ]

setup(
    name='py-earth',
    version='0.1.0',
    packages=['pyearth','pyearth.test'],
    py_modules = ['pyearth.earth'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    requires=['numpy','cython']
)
