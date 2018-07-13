#from distutils.core import setup
import setuptools
from Cython.Build import cythonize

setuptools.setup(
    name="steady",
    version="0.0.1",
    author="Jevon",
    author_email="jevon.longdell@gmail.com",
    install_requires=[
        'scipy',
        'cython'
        ],
    ext_modules=cythonize("steady.pyx"))
