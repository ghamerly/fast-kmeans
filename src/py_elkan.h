#ifndef PY_ELKAN_H
#define PY_ELKAN_H

#include <Python.h>
#include <structmember.h>

#include "elkan_kmeans.h"

typedef struct {
    PyObject_HEAD
    ElkanKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} ElkanObject;

extern PyTypeObject ElkanType;

void init_elkan_type_fields(void);

#endif
