import os
import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

def get_extensions():
    if os.name == "nt":
        compile_args = ['/std:c++17']
        link_args = ["psapi.lib"]
        libraries = []
    else:
        compile_args = ['-fopenmp', '-std=c++17']
        link_args = ['-fopenmp', '-std=c++17']
        libraries = ['m']

    extensions = [
        Extension(
            'GKM_',
            sources=['GKM_.pyx', 'CppFuns.cpp'],  
            language='c++',
            include_dirs=[numpy.get_include(),'.'],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            libraries=libraries
        )
    ]
    return extensions

if __name__ == '__main__':
    setup(
        name='GKM',
        ext_modules=cythonize(get_extensions(), compiler_directives={'language_level': 3}),
    )
