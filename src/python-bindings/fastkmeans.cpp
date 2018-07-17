/* Creates the fastkmeans Python module and adds the classes and module methods.
 * This module provides wrappers for the Kmeans algorithms in the fast-kmeans
 * library, as well as auxiliary classes.
 */

#include <Python.h> // Provides Python/C API 
#include <structmember.h> // Provides utilities for dealing with attributes

#include <cstring> // For strrchr

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
        PyModuleDef_HEAD_INIT, // m_base: required
        "fastkmeans", // m_name
        "Extension module for the fast-kmeans C++ library", // m_doc
        -1, // m_size
        Fastkmeans_methods, // m_methods
        NULL, // m_slots
        NULL, // m_traverse
        NULL, // m_clear
        NULL, // m_free
    };

    PyMODINIT_FUNC PyInit_fastkmeans(void) {
        // Initialize wrapper type structs
        for (int i = 0; type_object_ptrs[i] != NULL; i++) {
            if (PyType_Ready(type_object_ptrs[i]) < 0) {
                return NULL;
            }
        }

        // Initialize fastkmeans module
        PyObject *mod = PyModule_Create(&fastkmeansmodule);
        if (mod == NULL) {
            return NULL;
        }

        // Add each wrapper type object to the module
        for (int i = 0; type_object_ptrs[i] != NULL; i++) {
            // Extract class name from fully-qualified name
            const char *name = strrchr(type_object_ptrs[i]->tp_name, '.') + 1;
            Py_INCREF(type_object_ptrs[i]);
            PyModule_AddObject(mod, name, (PyObject *) type_object_ptrs[i]);
        }

        return mod;
    };
}
