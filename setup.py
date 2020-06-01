from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("evolve3D_2_C.pyx")
)