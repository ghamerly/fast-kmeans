#include "py_dataset.h"

// #include "dataset.h"

#include <climits>

/*
typedef struct {
    PyObject_HEAD
    Dataset *dataset;
} DatasetObject;
*/

static int Dataset_init(DatasetObject *self, PyObject *args, PyObject *kwargs) {
    // Dataset(aN, aD, keep_sds=False)

    int aN, aD, keepSDS = 0;
    char *kwlist[] = {"", "", "keep_sds", NULL};

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

/*
static PyObject * Dataset_get_nd(DatasetObject *self, void *closure) {
    return PyLong_FromLong(self->dataset->nd);
}

static int Datset_set_nd(DatasetObject *self, PyObject *value, void *closure) {
    PyErr_SetString(
}
*/

static PyGetSetDef Dataset_getsetters[] = {
    {"n", (getter) Dataset_get_n, (setter) Dataset_set_n, "number of records",
        NULL},
    {"d", (getter) Dataset_get_d, (setter) Dataset_set_d, "dimension",
        NULL},
    //{"nd", (getter) Dataset_get_nd, (setter) Dataset_set_nd, 
        //"shortcut for n * d", NULL},
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

// TODO add str (and repr?) methods to call Dataset::print w/ other ostream

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
    /*
    tp_name = "";
    tp_basicsize = 0;
    tp_itemsize = 0;

    tp_dealloc;
    tp_print;
    tp_getattr;
    tp_setattr;
    tp_as_sync;
    tp_repr;

    tp_as_number;
    tp_as_sequence;
    tp_as_mapping;

    tp_hash;
    tp_call;
    tp_str;
    tp_getattro;
    tp_setattro;

    tp_as_buffer;

    tp_flags;

    tp_doc;

    tp_traverse;

    tp_clear;

    tp_richcompare;

    tp_weaklistoffset;

    tp_iter;
    tp_iternext;

    tp_methods;
    tp_members;
    tp_getset;
    tp_base;
    tp_dict;
    tp_descr_get;
    tp_descr_set;
    tp_dictoffset;
    tp_init;
    tp_alloc;
    tp_new;
    tp_free;
    tp_is_gc;
    tp_bases;
    tp_mro;
    tp_cache;
    tp_subclasses;
    tp_weaklist;
    tp_del;

    tp_version_tag;
    tp_finalize;
    */
};

void init_dataset_type_fields(void) {
    DatasetType.tp_name = "fastkmeans.Dataset";
    DatasetType.tp_doc = "";
    DatasetType.tp_basicsize = sizeof(DatasetObject);
    DatasetType.tp_itemsize = 0;
    DatasetType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    DatasetType.tp_new = PyType_GenericNew;
    DatasetType.tp_init = (initproc) Dataset_init;
    DatasetType.tp_dealloc = (destructor) Dataset_dealloc;
    DatasetType.tp_methods = Dataset_methods;
    DatasetType.tp_getset = Dataset_getsetters;
    DatasetType.tp_as_mapping = &Dataset_mapping_methods;
}
