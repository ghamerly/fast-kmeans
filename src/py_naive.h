#ifndef PY_NAIVE_H
#define PY_NAIVE_H

#include <Python.h>
#include <structmember.h>

#include "naive_kmeans.h"

typedef struct {
    PyObject_HEAD
    NaiveKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} NaiveObject;

extern PyTypeObject NaiveType;

void init_naive_type_fields(void);

#endif
