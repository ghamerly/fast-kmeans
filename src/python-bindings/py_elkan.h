#ifndef PY_ELKAN_H
#define PY_ELKAN_H

/* Provides a wrapper for the ElkanKmeans class. See elkan_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "elkan_kmeans.h"

typedef struct {
    PyObject_HEAD
    ElkanKmeans *instance;
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} ElkanObject;

extern PyTypeObject ElkanType;

#endif
