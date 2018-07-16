#ifndef PY_NAIVE_H
#define PY_NAIVE_H

/* Provides a wrapper for the NaiveKmeans class. See naive_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "naive_kmeans.h"

typedef struct {
    PyObject_HEAD
    NaiveKmeans *instance;
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} NaiveObject;

extern PyTypeObject NaiveType;

#endif
