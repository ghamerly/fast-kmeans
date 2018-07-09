#ifndef PY_SORT_H
#define PY_SORT_H

#include <Python.h>
#include <structmember.h>

#include "sort_kmeans.h"

typedef struct {
    PyObject_HEAD
    SortKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} SortObject;

extern PyTypeObject SortType;

void init_sort_type_fields(void);

#endif
