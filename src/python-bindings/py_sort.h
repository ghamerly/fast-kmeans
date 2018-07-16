#ifndef PY_SORT_H
#define PY_SORT_H

/* Provides a wrapper for the SortKmeans class. See sort_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "sort_kmeans.h"

typedef struct {
    PyObject_HEAD
    SortKmeans *instance;
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} SortObject;

extern PyTypeObject SortType;

#endif
