#ifndef PY_COMPARE_H
#define PY_COMPARE_H

/* Provides a wrapper for the CompareKmeans class. See compare_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "compare_kmeans.h"

typedef struct {
    PyObject_HEAD
    CompareKmeans *instance;
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} CompareObject;

extern PyTypeObject CompareType;

#endif
