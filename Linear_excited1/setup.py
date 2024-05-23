from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules = cythonize("c_linear_excited1.pyx"))
setup(ext_modules = cythonize("c_linear_excited2.pyx"))
setup(ext_modules = cythonize("c_linear_excited1big.pyx"))
setup(ext_modules = cythonize("c_linear_excited3.pyx"))
# setup(ext_modules = cythonize("c_linear_excited4.pyx"))
