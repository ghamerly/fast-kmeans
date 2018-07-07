#ifndef PY_COMPARE_H
#define PY_COMPARE_H

#include <Python.h>
#include <structmember.h>

#include "compare_kmeans.h"

typedef struct {
    PyObject_HEAD
    CompareKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} CompareObject;

extern PyTypeObject CompareType;

void init_compare_type_fields(void);

#endif
