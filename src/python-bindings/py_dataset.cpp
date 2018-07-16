/* Dataset wrapper. The comment at the beginning of each function definition
 * demonstrates its usage in Python.
 */

#include "py_dataset.h"

// #include "dataset.h"

#include <climits>
#include <sstream>

/*
typedef struct {
    PyObject_HEAD
    Dataset *dataset;
} DatasetObject;
*/

static int Dataset_init(DatasetObject *self, PyObject *args, PyObject *kwargs) {
    // Dataset(aN, aD, keep_sds=False)

    int aN, aD, keepSDS = 0;
    char *emptyStr = const_cast<char *>("");
    char *kwlist[] = {emptyStr, emptyStr, const_cast<char *>("keep_sds"), NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|p", kwlist, &aN, &aD,
                &keepSDS)) {
        return -1;
    }

    self->dataset = new Dataset(aN, aD, keepSDS);

    return 0;
}

static void Dataset_dealloc(DatasetObject *self) {
    delete self->dataset;
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject * Dataset_get_n(DatasetObject *self, void *closure) {
    // a_dataset.n

    return PyLong_FromLong(self->dataset->n);
}

static int Dataset_set_n(DatasetObject *self, PyObject *value, void *closure) {
    // a_dataset.n = some_val

    if (PyLong_Check(value)) {
        long newN = PyLong_AsLong(value);

        if (PyErr_Occurred() == NULL && newN >= 0 && newN <= INT_MAX) {
            self->dataset->n = newN;
        } else {
            if (PyErr_Occurred() == NULL) {
                PyErr_SetString(PyExc_ValueError, "n must be a positive int");
            }
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "n must be a positive int");
        return -1;
    }

    return 0;
}

static PyObject * Dataset_get_d(DatasetObject *self, void *closure) {
    // a_dataset.d

    return PyLong_FromLong(self->dataset->d);
}

static int Dataset_set_d(DatasetObject *self, PyObject *value, void *closure) {
    // a_dataset.d = some_val

    if (PyLong_Check(value)) {
        long newD = PyLong_AsLong(value);

        if (PyErr_Occurred() == NULL && newD >= 0 && newD <= INT_MAX) {
            self->dataset->d = newD;
        } else {
            if (PyErr_Occurred() == NULL) {
                PyErr_SetString(PyExc_ValueError, "d must be a positive int");
            }
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "d must be a positive int");
        return -1;
    }

    return 0;
}

static PyGetSetDef Dataset_getsetters[] = {
    {const_cast<char *>("n"), (getter) Dataset_get_n, (setter) Dataset_set_n, const_cast<char *>("number of records"),
        NULL},
    {const_cast<char *>("d"), (getter) Dataset_get_d, (setter) Dataset_set_d, const_cast<char *>("dimension"),
        NULL},
    {NULL} // Sentinel
};

static PyObject * Dataset_fill(DatasetObject *self, PyObject *o) {
    // a_dataset.fill(a_double)

    double value = 0.0;

    // Check whether a numeric type
    if (PyFloat_Check(o)) {
        value = PyFloat_AsDouble(o);
    } else if (PyLong_Check(o)) {
        value = PyLong_AsDouble(o);
    } else {
        PyErr_SetString(PyExc_TypeError, "must fill with a double-precision "
                "float value");
    }

    // Exit if type invalid or error occurred during type conversion
    if (PyErr_Occurred() != NULL) {
        return NULL;
    }

    self->dataset->fill(value);

    Py_RETURN_NONE;
}

static PyObject * Dataset_print(DatasetObject *self) {
    // a_dataset.print()

    self->dataset->print();
    Py_RETURN_NONE;
}

static PyMethodDef Dataset_methods[] = {
    {"fill", (PyCFunction) Dataset_fill, METH_O, "Fill the entire "
        "dataset with value. Does NOT update sumDataSquared."},
    {"print", (PyCFunction) Dataset_print, METH_NOARGS, "Print to std out "
        "in matrix format"},
    {NULL} // Sentinel
};

static PyObject * Dataset_str(DatasetObject *self) {
    // print(a_dataset)

    std::stringstream out;
    self->dataset->print(out);
    return PyUnicode_FromString(out.str().c_str());
}

static Py_ssize_t Dataset_len(DatasetObject *self) {
    // len(a_dataset)

    return self->dataset->n;
}

static PyObject * Dataset_subscript(DatasetObject *self, PyObject *key) {
    // a_dataset[i,j]

    double val = 0.0;

    if (PyTuple_Check(key) && PyTuple_Size(key) == 2
            && PyLong_Check(PyTuple_GetItem(key, 0))
            && PyLong_Check(PyTuple_GetItem(key, 1))) {
        long i = PyLong_AsLong(PyTuple_GetItem(key, 0)),
             j = PyLong_AsLong(PyTuple_GetItem(key, 1));
            
        if (0 <= i && i < self->dataset->n && 0 <= j && j < self->dataset->d) {
            // Retrieve the value at that point and dim
            val = (*(self->dataset))(i, j);
        } else {
            PyErr_SetString(PyExc_KeyError, "keys must be less than n, d");
            PyErr_SetString(PyExc_KeyError, "key values must be in range [0,n) "
                        "and [0,d), respectively");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be tuple of two ints");
        return NULL;
    }

    return PyFloat_FromDouble(val);
}

static int Dataset_ass_subscript(DatasetObject *self, PyObject *key,
        PyObject *v) {
    // a_dataset[i,j] = some_val

    if (PyTuple_Check(key) && PyTuple_Size(key) == 2
            && PyLong_Check(PyTuple_GetItem(key, 0))
            && PyLong_Check(PyTuple_GetItem(key, 1))) {
        long i = PyLong_AsLong(PyTuple_GetItem(key, 0)),
             j = PyLong_AsLong(PyTuple_GetItem(key, 1));
            
        if (0 <= i && i < self->dataset->n && 0 <= j && j < self->dataset->d) {
            double val = -1.0;

            if (PyFloat_Check(v)) {
                val = PyFloat_AsDouble(v);
            } else if (PyLong_Check(v)) {
                val = PyLong_AsDouble(v);
            } else {
                PyErr_SetString(PyExc_TypeError, "v must be a real number");
                return -1;
            }

            if (val == -1.0 && PyErr_Occurred() != NULL) {
                return -1;
            }

            // Set the value at the specified point and dim
            (*(self->dataset))(i, j) = val;
        } else {
            PyErr_SetString(PyExc_KeyError, "keys must be less than n, d");
            PyErr_SetString(PyExc_KeyError, "key values must be in range [0,n) "
                        "and [0,d), respectively");
            return -1;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "key must be tuple of two ints");
        return -1;
    }

    return 0;
}

static PyMappingMethods Dataset_mapping_methods = {
    (lenfunc) Dataset_len,
    (binaryfunc) Dataset_subscript,
    (objobjargproc) Dataset_ass_subscript
};

PyTypeObject DatasetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fastkmeans.Dataset", // tp_name = ""
    sizeof(DatasetObject), // tp_basicsize = 0
    0, // tp_itemsize = 0

    (destructor) Dataset_dealloc, // tp_dealloc
    NULL, // tp_print
    NULL, // tp_getattr
    NULL, // tp_setattr
    NULL, // tp_as_sync
    NULL, // tp_repr

    NULL, // tp_as_number
    NULL, // tp_as_sequence
    &Dataset_mapping_methods, // tp_as_mapping

    NULL, // tp_hash
    NULL, // tp_call TODO ?
    (reprfunc) Dataset_str, // tp_str
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

    Dataset_methods, // tp_methods
    NULL, // tp_members
    Dataset_getsetters, // tp_getset
    NULL, // tp_base
    NULL, // tp_dict
    NULL, // tp_descr_get
    NULL, // tp_descr_set
    0, // tp_dictoffset
    (initproc) Dataset_init, // tp_init
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
