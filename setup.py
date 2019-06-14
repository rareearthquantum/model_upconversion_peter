from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('c_funs_excited2.pyx'))

setup(ext_modules=cythonize('c_funs_ds.pyx'))
setup(ext_modules=cythonize('c_funs_excited_ds.pyx'))
setup(ext_modules=cythonize('c_funs_ds2.pyx'))
setup(ext_modules=cythonize('c_funs_test4.pyx'))
