from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
libs_path='-I/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/include -I/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/numpy/core/include -Ibuild/src.macosx-10.14-x86_64-3.9/numpy/distutils/include -I/usr/local/include -I/usr/local/opt/openssl@1.1/include -I/usr/local/opt/sqlite/include -I/usr/local/lib/python3.9/site-packages/numpy/core/include -I/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/include/python3.9 -c'

setup(
    ext_modules=cythonize('_navigable_small_world_graph.pyx',
                          compiler_directives={'language_level':"3"},
                          include_path = [libs_path]),
)

"""
clang++ -bundle -undefined dynamic_lookup -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk -I/usr/local/lib/python3.9/site-packages/numpy/core/include build/temp.macosx-10.14-x86_64-3.9/_navigable_small_world_graph.o -L/usr/local/lib -L/usr/local/opt/openssl@1.1/lib -L/usr/local/opt/sqlite/lib  -o _navigable_small_world_graph.so
"""