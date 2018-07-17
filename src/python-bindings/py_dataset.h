#ifndef PY_DATASET_H
#define PY_DATASET_H

/* Provides a wrapper for the Dataset class. See dataset.h for more detail.
 */

#include <Python.h>
#include <structmember.h>

#include "dataset.h"

typedef struct {
    PyObject_HEAD
    Dataset *dataset;
} DatasetObject;

extern PyTypeObject DatasetType;

#endif
