#from distutils.core import setup
import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize


extensions = [
    Extension("steady", ["steady.pyx"],
#        include_dirs = [...],
        libraries = ['lapacke']
#        library_dirs = [...]
    )
    # Everything but primes.pyx is included here.
]


setuptools.setup(
    name="steady",
    version="0.0.1",
    author="Jevon",
    author_email="jevon.longdell@gmail.com",
    install_requires=[
        'scipy',
        'cython'
        ],
    ext_modules=cythonize(extensions)
    )
