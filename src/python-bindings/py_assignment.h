#ifndef PY_ASSIGNMENT_H
#define PY_ASSIGNMENT_H

/* Provides a wrapper of an array of unsigned shorts. This allows for better
 * performance than having to access the elements of a more native Python
 * sequence, like lists or tuples, while still providing a sequence interface.
 * Assignment instances are returned by the fastkmeans.assign method and can be
 * passed to the [kmeans_algorithm].initialize methods, so the user should not
 * have to deal with it in depth under normal circumstances.
 */

#include <Python.h>
#include <structmember.h>

typedef struct {
    PyObject_HEAD
    int n;
    unsigned short *assignment;
} AssignmentObject;

extern PyTypeObject AssignmentType;

#endif
