#include "py_assignment.h"

#include <climits>

/*
typedef struct {
    PyObject_HEAD
    int n;
    unsigned short *assignment;
} AssignmentObject;
*/

static int Assignment_init(AssignmentObject *self, PyObject *args) { //, PyObject *kwargs) {
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

// static PyObject * Assignment_get_n(AssignmentObject *self, void *closure) {
    // Assignment.n
    // return PyLong_FromLong(self->n);
// }

// static PyGetSetDef Assignment_getsetters[] = {
    // {"n", (getter) Assignment_get_n, NULL, "number of records", NULL},
    // {NULL} // Sentinel
// };

static PyMemberDef Assignment_members[] = {
    {"n", T_INT, offsetof(AssignmentObject, n), READONLY, "The length of the "
        "array of unsigned shorts"},
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

// TODO add str (and repr?) methods to call Assignment::print w/ other ostream

static PyMethodDef Assignment_methods[] = {
    {"fill", (PyCFunction) Assignment_fill, METH_O, "Fill the entire "
        "assignment with value"},
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
    NULL        // sq_inplace_repeat
};

PyTypeObject AssignmentType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

void init_assignment_type_fields(void) {
    AssignmentType.tp_name = "fastkmeans.Assignment";
    AssignmentType.tp_doc = "";
    AssignmentType.tp_basicsize = sizeof(AssignmentObject);
    AssignmentType.tp_itemsize = 0;
    AssignmentType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    AssignmentType.tp_new = PyType_GenericNew;
    AssignmentType.tp_init = (initproc) Assignment_init;
    AssignmentType.tp_dealloc = (destructor) Assignment_dealloc;
    AssignmentType.tp_methods = Assignment_methods;
    AssignmentType.tp_members = Assignment_members;
    // AssignmentType.tp_getset = Assignment_getsetters;
    AssignmentType.tp_as_sequence = &Assignment_sequence_methods;
}
