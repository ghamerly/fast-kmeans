#ifndef PY_ANNULUS_H
#define PY_ANNULUS_H

/* Provides a wrapper for the AnnulusKmeans class. See annulus_kmeans.h for more
 * detail.
 */

#include <Python.h>
#include <structmember.h>

#include "annulus_kmeans.h"

typedef struct {
    PyObject_HEAD
    AnnulusKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} AnnulusObject;

extern PyTypeObject AnnulusType;

#endif
