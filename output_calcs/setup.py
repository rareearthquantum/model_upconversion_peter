from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('c_funs_test3.pyx'))
#setup(ext_modules=cythonize('c_funs_ds.pyx'))
