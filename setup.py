from setuptools import Extension
from distutils.core import setup
import numpy
from Cython.Build import cythonize
# libs_path = '/usr/local/lib/python3.9/dist-packages/numpy/core/include'
setup(
    ext_modules=cythonize(
        Extension('_navigable_small_world_graph',
                  ['_navigable_small_world_graph.pyx'],
                  extra_compile_args=["-fopenmp"],
                  extra_link_args=["-fopenmp"]
                  ),
                  compiler_directives={'language_level': "3"},
                  nthreads=10,
                  quiet=True,
                  # annotate=True
                  ),include_dirs=[numpy.get_include()],
)