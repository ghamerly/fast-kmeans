#ifndef PY_HAMERLY_H
#define PY_HAMERLY_H

#include <Python.h>
#include <structmember.h>

#include "hamerly_kmeans.h"

typedef struct {
    PyObject_HEAD
    HamerlyKmeans *instance;
    // TODO make these visible, probably readonly
    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} HamerlyObject;

extern PyTypeObject HamerlyType;

void init_hamerly_type_fields(void);

#endif
