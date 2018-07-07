#ifndef PY_DRAKE_H
#define PY_DRAKE_H

#include <Python.h>
#include <structmember.h>

#include "drake_kmeans.h"

typedef struct {
    PyObject_HEAD
    DrakeKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} DrakeObject;

extern PyTypeObject DrakeType;

void init_drake_type_fields(void);

#endif
