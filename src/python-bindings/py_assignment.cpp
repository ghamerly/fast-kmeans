/* Assignment wrapper. The comment at the beginning of each function
 * definition demonstrates its usage in Python.
 */

#include "py_assignment.h"

#include <climits>

/*
typedef struct {
    PyObject_HEAD
    int n;
    unsigned short *assignment;
} AssignmentObject;
*/

static int Assignment_init(AssignmentObject *self, PyObject *args) {
    // Assignment(n)

    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) {
        return -1;
    }

    self->n = n;
    self->assignment = new unsigned short[n];

    return 0;
}

static void Assignment_dealloc(AssignmentObject *self) {
    delete self->assignment;
    self->n = 0;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyMemberDef Assignment_members[] = {
    {
        const_cast<char *>("n"),
        T_INT,
        offsetof(AssignmentObject, n),
        READONLY,
        const_cast<char *>("The length of the array of unsigned shorts"),
    },
    {NULL} // Sentinel
};

static PyObject * Assignment_fill(AssignmentObject *self, PyObject *o) {
    // an_assignment.fill(an_unsigned_short)

    long value = 0;

    // Check whether a valid unsigned short value
    if (PyLong_Check(o)) {
        value = PyLong_AsLong(o);

        if (value > USHRT_MAX) {
            PyErr_SetString(PyExc_ValueError, "value must be an unsigned short");
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "value must be an unsigned short");
    }

    // Exit if type invalid or error occurred during type conversion
    if (PyErr_Occurred() != NULL) {
        return NULL;
    }

    for (int i = 0; i < self->n; i++) {
        self->assignment[i] = value;
    }

    Py_RETURN_NONE;
}

static PyMethodDef Assignment_methods[] = {
    {
        const_cast<char *>("fill"),
        (PyCFunction) Assignment_fill,
        METH_O,
        const_cast<char *>("Fill the entire assignment with value"),
    },
    {NULL} // Sentinel
};

static Py_ssize_t Assignment_len(AssignmentObject *self) {
    // len(an_assignment)

    return self->n;
}

static PyObject * Assignment_item(AssignmentObject *self, Py_ssize_t i) {
    // an_assignment[i]

    // .sq_length is defined, so any int index in range [-1,-n] will be made
    // positive, but other negative indices will still be negative

    if (self->n <= 0 || self->assignment == NULL) {
        PyErr_SetString(PyExc_IndexError, "cannot index an empty sequence");
        return NULL;
    } else if (i < 0 || i >= self->n) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return NULL;
    }

    return PyLong_FromUnsignedLong(self->assignment[i]);
}

static int Assignment_ass_item(AssignmentObject *self, Py_ssize_t i,
        PyObject *val) {
    // an_assignment[i] = an_integer_val

    // .sq_length is defined, so any int index in range [-1,-n] will be made
    // positive, but other negative indices will still be negative

    if (self->n <= 0 || self->assignment == NULL) {
        PyErr_SetString(PyExc_IndexError, "cannot index an empty sequence");
        return -1;
    } else if (i < 0 || i >= self->n) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        return -1;
    }

    long longVal = PyLong_AsLong(val);
    if (PyErr_Occurred()) {
        return -1;
    } else if (longVal > USHRT_MAX) {
        PyErr_SetString(PyExc_ValueError, "value must be an unsigned short");
        return -1;
    }

    self->assignment[i] = longVal;

    return 0;
}

static PySequenceMethods Assignment_sequence_methods = {
    (lenfunc) Assignment_len,
    NULL,       // sq_concat
    NULL,       // sq_repeat
    (ssizeargfunc) Assignment_item,
    NULL,       // sq_slice
    (ssizeobjargproc) Assignment_ass_item,
    NULL,       // sq_ass_slice
    NULL,       // sq_contains
    NULL,       // sq_inplace_concat
    NULL,       // sq_inplace_repeat
};

PyTypeObject AssignmentType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fastkmeans.Assignment", // tp_name
    sizeof(AssignmentObject), // tp_basicsize
    0, // tp_itemsize

    (destructor) Assignment_dealloc, // tp_dealloc
    NULL, // tp_print
    NULL, // tp_getattr
    NULL, // tp_setattr
    NULL, // tp_as_sync
    NULL, // tp_repr

    NULL, // tp_as_number
    &Assignment_sequence_methods, // tp_as_sequence
    NULL, // tp_as_mapping

    NULL, // tp_hash
    NULL, // tp_call TODO ?
    NULL, // tp_str
    NULL, // tp_getattro
    NULL, // tp_setattro

    NULL, // tp_as_buffer

    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // tp_flags

    "", // tp_doc

    NULL, // tp_traverse

    NULL, // tp_clear

    NULL, // tp_richcompare

    0, // tp_weaklistoffset

    NULL, // tp_iter
    NULL, // tp_iternext

    Assignment_methods, // tp_methods
    Assignment_members, // tp_members
    NULL, // tp_getset
    NULL, // tp_base
    NULL, // tp_dict
    NULL, // tp_descr_get
    NULL, // tp_descr_set
    0, // tp_dictoffset
    (initproc) Assignment_init, // tp_init
    PyType_GenericAlloc, // tp_alloc
    PyType_GenericNew, // tp_new
    NULL, // tp_free
    NULL, // tp_is_gc
    NULL, // tp_bases
    NULL, // tp_mro
    NULL, // tp_cache
    NULL, // tp_subclasses
    NULL, // tp_weaklist
    NULL, // tp_del

    0, // tp_version_tag
    NULL, // tp_finalize
};
