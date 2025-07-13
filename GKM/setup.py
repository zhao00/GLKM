# import os
# import numpy
# from Cython.Build import cythonize

# def configuration(parent_package='', top_path=None):
#     from numpy.distutils.misc_util import Configuration

#     cpp_version = "c++17"
#     if os.name == "nt":
#         ext_comp_args = ['/']
#         ext_link_args = []

#         library_dirs = []
#         libraries = []
#     else:
#         ext_comp_args = ['-fopenmp', f'-std={cpp_version}']
#         ext_link_args = ['-fopenmp', f'-std={cpp_version}']
#         library_dirs = []
#         libraries = ["m"]

#     fpath, fname = os.path.split(os.path.abspath(__file__))

#     config = Configuration('HKM_new', parent_package, top_path)

#     config.add_extension('HKM_',
#                          sources=['HKM_.pyx'],
#                          include_dirs=[numpy.get_include()],
#                          language="c++",
#                          libraries=libraries)

#     config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

#     return config


# if __name__ == '__main__':
#     #from numpy.distutils.core import setup
#     from setuptools import setup
#     setup(**configuration(top_path='').todict())

# # python setup.py build_ext --inplace


# import os
# import numpy
# from Cython.Build import cythonize
# from setuptools import setup, Extension

# def configuration():
#     cpp_version = "c++17"
#     define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
#     if os.name == "nt":
#         ext_comp_args = ['/openmp']
#         ext_link_args = ['/openmp']

#         # library_dirs = [cg.mpir_path]     
#         libraries = []
#     else:
#         ext_comp_args = ['-fopenmp', f'-std={cpp_version}']
#         ext_link_args = ['-fopenmp', f'-std={cpp_version}']
#         libraries = ["m"]

#     extensions = [
#         Extension(
#             'GKM_',
#             sources=['GKM_.pyx'],
#             include_dirs=[numpy.get_include()],
#             language="c++",
#             extra_compile_args=ext_comp_args,
#             extra_link_args=ext_link_args,
#             libraries=libraries
#         )
#     ]

#     return extensions

# if __name__ == '__main__':
#     setup(
#         name='GKM',
#         ext_modules=cythonize(configuration(), compiler_directives={'language_level': 3}),
#     )


import os
import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

def get_extensions():
    if os.name == "nt":
        # MSVC 编译器下的 C++17 写法
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
            sources=['GKM_.pyx', 'CppFuns.cpp'],  # 加入 .cpp 文件如果有
            language='c++',
            include_dirs=[numpy.get_include(),'CppFuns.h'],
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
