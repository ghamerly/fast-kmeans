#ifndef PY_ASSIGNMENT_H
#define PY_ASSIGNMENT_H

#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    int n;
    unsigned short *assignment;
} AssignmentObject;

extern PyTypeObject AssignmentType;

void init_assignment_type_fields(void);

#endif
