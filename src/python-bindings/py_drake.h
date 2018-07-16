#ifndef PY_DRAKE_H
#define PY_DRAKE_H

/* Provides a wrapper for the DrakeKmeans class. See drake_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "drake_kmeans.h"

typedef struct {
    PyObject_HEAD
    DrakeKmeans *instance;
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} DrakeObject;

extern PyTypeObject DrakeType;

#endif
