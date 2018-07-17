#ifndef PY_HEAP_H
#define PY_HEAP_H

/* Provides a wrapper for the HeapKmeans class. See heap_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "heap_kmeans.h"

typedef struct {
    PyObject_HEAD
    HeapKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} HeapObject;

extern PyTypeObject HeapType;

#endif
