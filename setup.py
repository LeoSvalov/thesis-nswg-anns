from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

# libs_path = '/usr/local/lib/python3.9/dist-packages/numpy/core/include' 
setup(
    ext_modules=cythonize('_navigable_small_world_graph.pyx',
                          compiler_directives={'language_level':"3"}),
)

