from setuptools import setup, Extension
import sys
import os
import codecs
sys.path.insert(0, os.path.join('.', 'pyearth'))
from _version import __version__

# Determine whether to use Cython
if '--cythonize' in sys.argv:
    cythonize_switch = True
    del sys.argv[sys.argv.index('--cythonize')]
else:
    cythonize_switch = False


def get_ext_modules():
    import numpy
    # Find all includes
    local_inc = 'pyearth'
    numpy_inc = numpy.get_include()

    # Set up the ext_modules for Cython or not, depending
    if cythonize_switch:
        from Cython.Build import cythonize
        ext_modules = cythonize(
            [Extension(
                "pyearth._util", ["pyearth/_util.pyx"], include_dirs=[numpy_inc]),
             Extension(
                 "pyearth._basis",
                 ["pyearth/_basis.pyx"],
                 include_dirs=[numpy_inc]),
             Extension(
                 "pyearth._record",
                 ["pyearth/_record.pyx"],
                 include_dirs=[numpy_inc]),
             Extension(
                 "pyearth._pruning",
                 ["pyearth/_pruning.pyx"],
                 include_dirs=[local_inc,
                               numpy_inc]),
             Extension(
                 "pyearth._forward",
                 ["pyearth/_forward.pyx"],
                 include_dirs=[local_inc,
                               numpy_inc]),
             Extension(
                 "pyearth._types",
                 ["pyearth/_types.pyx"],
                 include_dirs=[local_inc,
                               numpy_inc])
             ])
    else:
        ext_modules = [Extension(
            "pyearth._util", ["pyearth/_util.c"], include_dirs=[numpy_inc]),
            Extension(
                "pyearth._basis",
                ["pyearth/_basis.c"],
                include_dirs=[numpy_inc]),
            Extension(
                "pyearth._record",
                ["pyearth/_record.c"],
                include_dirs=[numpy_inc]),
            Extension(
                "pyearth._pruning",
                ["pyearth/_pruning.c"],
                include_dirs=[local_inc,
                              numpy_inc]),
            Extension(
                "pyearth._forward",
                ["pyearth/_forward.c"],
                include_dirs=[local_inc,
                              numpy_inc]),
            Extension(
                "pyearth._types",
                ["pyearth/_types.c"],
                include_dirs=[local_inc,
                              numpy_inc])
        ]
    return ext_modules

def setup_package():
    # Create a dictionary of arguments for setup
    setup_args = {
        'name': 'py-earth',
        'version': __version__,
        'author': 'Jason Rudy',
        'author_email': 'jcrudy@gmail.com',
        'packages': ['pyearth', 'pyearth.test',
                   'pyearth.test.basis', 'pyearth.test.record'],
        'license': 'LICENSE.txt',
        'description':
        'A Python implementation of Jerome Friedman\'s MARS algorithm.',
        'long_description': codecs.open('README.md', mode='r', encoding='utf-8').read(),
        'py_modules': ['pyearth.earth', 'pyearth._version'],
        'classifiers': ['Development Status :: 3 - Alpha'],
        'requires': ['numpy', 'scipy'],
        'install_requires': ['scikit-learn >= 0.16',
                           'sphinx_gallery'],
        'setup_requires': ['numpy']
    }

    # Add the build_ext command only if cythonizing
    if cythonize_switch:
        from Cython.Distutils import build_ext
        setup_args['cmdclass'] = {'build_ext': build_ext}

    
    def is_special_command():
        special_list = ('--help-commands', 
                        'egg_info',  
                        '--version',
                        'clean')
        return ('--help' in sys.argv[1:] or 
                sys.argv[1] in special_list)

    if len(sys.argv) >= 2 and is_special_command():
        setup(**setup_args)
    else:
        setup_args['ext_modules'] = get_ext_modules()
        setup(**setup_args)

if __name__ == "__main__":
    setup_package()
