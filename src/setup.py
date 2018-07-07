from distutils.core import setup, Extension

setup(
    name='fastkmeans',
    version='0.1',
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
            include_dirs = ['.'], # Useful if project reorganized
            library_dirs=['.'],
            libraries=['kmeans'],
        ),
    ]
)
