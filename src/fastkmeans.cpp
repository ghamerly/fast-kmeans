#include <Python.h> // Provides Python/C API 
#include <structmember.h> // Provides utilities for dealing with attributes

#include "py_annulus.h"
#include "py_assignment.h"
#include "py_compare.h"
#include "py_dataset.h"
#include "py_drake.h"
#include "py_elkan.h"
#include "py_hamerly.h"
#include "py_heap.h"
#include "py_fastkmeans_methods.h"
#include "py_naive.h"
#include "py_sort.h"

#include <cstring>

extern "C" {
    static PyTypeObject *type_object_ptrs[] = {
        &AnnulusType,
        &AssignmentType,
        &CompareType,
        &DatasetType,
        &DrakeType,
        &ElkanType,
        &HamerlyType,
        &HeapType,
        &NaiveType,
        &SortType,
        NULL
    };

    static PyModuleDef fastkmeansmodule = {
        PyModuleDef_HEAD_INIT, // .m_base: required
        "fastkmeans", // .m_name
        "Extension module for the fast-kmeans C++ library", // .m_doc
        -1, // .m_size
        Fastkmeans_methods, // .m_methods
        // .m_slots
        // .m_traverse
        // .m_clear
        // .m_free
    };

    PyMODINIT_FUNC PyInit_fastkmeans(void) {
        // TODO make each of these return PyTypeObject * instead?

        init_annulus_type_fields();
        init_assignment_type_fields();
        init_compare_type_fields();
        init_dataset_type_fields();
        init_drake_type_fields();
        init_elkan_type_fields();
        init_hamerly_type_fields();
        init_heap_type_fields();
        init_naive_type_fields();
        init_sort_type_fields();

        for (int i = 0; type_object_ptrs[i] != NULL; i++) {
            if (PyType_Ready(type_object_ptrs[i]) < 0) {
                return NULL;
            }
        }

        PyObject *mod = PyModule_Create(&fastkmeansmodule);
        if (mod == NULL) {
            return NULL;
        }

        for (int i = 0; type_object_ptrs[i] != NULL; i++) {
            // Extract class name from fully-qualified name
            const char *name = strrchr(type_object_ptrs[i]->tp_name, '.') + 1;
            Py_INCREF(type_object_ptrs[i]);
            PyModule_AddObject(mod, name, (PyObject *) type_object_ptrs[i]);
        }

        return mod;
    };
}
