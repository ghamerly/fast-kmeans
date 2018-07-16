#!/usr/bin/env python3

'''Builds the fastkmeans Python extension module.
'''

### Imports ###

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler

### Classes ###


class my_build_ext(build_ext):
    '''This is necessary to remove a useless compiler warning. Code taken from
    this solution:

    https://stackoverflow.com/a/36293331/9201637
    '''

    def build_extensions(self):
        # Perform any platform-specific customization
        customize_compiler(self.compiler)
        try:
            # Remove specific compiler option
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            # If not in list of options, the problem is already solved
            pass

        build_ext.build_extensions(self)


### Main ###

setup(
    name='fastkmeans',
    version='0.1',
    cmdclass={'build_ext': my_build_ext},
    ext_modules=[
        Extension(
            'fastkmeans',
            [
                'fastkmeans.cpp',
                'py_annulus.cpp',
                'py_assignment.cpp',
                'py_compare.cpp',
                'py_dataset.cpp',
                'py_drake.cpp',
                'py_elkan.cpp',
                'py_hamerly.cpp',
                'py_heap.cpp',
                'py_fastkmeans_methods.cpp',
                'py_naive.cpp',
                'py_sort.cpp',
            ],
            include_dirs = ['..'],
            library_dirs=['..'],
            libraries=['kmeans'],
        ),
    ]
)
