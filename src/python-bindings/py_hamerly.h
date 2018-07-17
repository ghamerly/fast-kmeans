#ifndef PY_HAMERLY_H
#define PY_HAMERLY_H

/* Provides a wrapper for the HamerlyKmeans class. See hamerly_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "hamerly_kmeans.h"

typedef struct {
    PyObject_HEAD
    HamerlyKmeans *instance;
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} HamerlyObject;

extern PyTypeObject HamerlyType;

#endif
