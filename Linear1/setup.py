from distutils.core import setup
from Cython.Build import cythonize


setup(ext_modules = cythonize("c_linear_ground2.pyx"))
setup(ext_modules = cythonize("c_linear_ground1.pyx"))
setup(ext_modules = cythonize("c_linear_ground3.pyx"))
setup(ext_modules = cythonize("c_linear_ground4.pyx"))
setup(ext_modules = cythonize("c_linear_ground_testbounds.pyx"))
setup(ext_modules = cythonize("c_linear_ground_slow_af.pyx"))
setup(ext_modules = cythonize("c_linear_ground5.pyx"))
setup(ext_modules = cythonize("c_linear_ground5big.pyx"))
setup(ext_modules = cythonize("c_linear_ground5BAD.pyx"))
