#ifndef PY_DATASET_H
#define PY_DATASET_H

#include <Python.h>
#include <structmember.h>

#include "dataset.h"

typedef struct {
    PyObject_HEAD
    Dataset *dataset;
} DatasetObject;

extern PyTypeObject DatasetType;

void init_dataset_type_fields(void);

#endif
