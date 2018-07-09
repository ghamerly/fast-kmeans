#include "py_fastkmeans_methods.h"

#include "py_assignment.h"
#include "py_dataset.h"

#include "general_functions.h"

// addVectors

// subVectors

// distance2silent

static PyObject * Fastkmeans_center_dataset(PyObject *self, PyObject *args) {
    // center_dataset(a_dataset)

    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O!", &DatasetType, &obj)) {
        return NULL;
    }

    DatasetObject *d = (DatasetObject *) obj;
    centerDataset(d->dataset);

    Py_RETURN_NONE;
}

static PyObject * init_centers_with_func(PyObject *self, PyObject *args,
        Dataset * (*init_func)(Dataset const &x, unsigned short k)) {
    PyObject *obj;
    unsigned short k;

    if (!PyArg_ParseTuple(args, "O!H", &DatasetType, &obj, &k)) {
        return NULL;
    }

    DatasetObject *d = (DatasetObject *) obj;
    Dataset *centers = init_func(*(d->dataset), k);

    PyObject *val = Py_BuildValue("ii", centers->n, centers->d);
    DatasetObject *c = (DatasetObject *)
        PyObject_CallObject((PyObject *) &DatasetType, val);
    c->dataset = centers;

    return (PyObject *) c;
}

static PyObject * Fastkmeans_init_centers_rand(PyObject *self, PyObject *args) {
    // init_centers_rand(a_dataset, k)

    return init_centers_with_func(self, args, init_centers);
}

static PyObject * Fastkmeans_kmeans_plusplus(PyObject *self,
        PyObject *args) {
    // kmeans_plusplus(a_dataset, k)

    return init_centers_with_func(self, args, init_centers_kmeanspp);
}

static PyObject * Fastkmeans_kmeans_plusplus_v2(PyObject *self,
        PyObject *args) {
    // kmeans_plusplus_v2(a_dataset, k)

    return init_centers_with_func(self, args, init_centers_kmeanspp_v2);
}

static PyObject * Fastkmeans_get_memory_usage(PyObject *self) {
    // get_memory_usage()

    return PyFloat_FromDouble(getMemoryUsage());
}

static PyObject * Fastkmeans_assign(PyObject *self, PyObject *args) {
    // assign(dataset, centers, assignment)

    PyObject *x, *c, *a;
    if (!PyArg_ParseTuple(args, "O!O!O!", &DatasetType, &x, &DatasetType, &c,
                &AssignmentType, &a)) {
        return NULL;
    }

    DatasetObject *dataset = (DatasetObject *) x;
    DatasetObject *centers = (DatasetObject *) c;
    AssignmentObject *assignment = (AssignmentObject *) a;

    assign(*(dataset->dataset), *(centers->dataset), assignment->assignment);

    Py_RETURN_NONE;
}

// get_time

// get_wall_time

// timeval_subtract

// elapsed_time

PyMethodDef Fastkmeans_methods[] = {
    {"center_dataset", (PyCFunction) Fastkmeans_center_dataset, METH_VARARGS,
        ""},
    {"init_centers_rand", (PyCFunction) Fastkmeans_init_centers_rand,
        METH_VARARGS, 
        "Initialize the centers randomly. Choose random records from x as "
        "the initial values for the centers. Assumes that c uses the "
        "sumDataSquared field."},
    {"kmeans_plusplus", (PyCFunction) Fastkmeans_kmeans_plusplus, METH_VARARGS, 
        "Initialize the centers randomly using K-means++."},
    {"kmeans_plusplus_v2", (PyCFunction) Fastkmeans_kmeans_plusplus_v2,
        METH_VARARGS, "Initialize the centers randomly using K-means++."},
    {"get_memory_usage", (PyCFunction) Fastkmeans_get_memory_usage, METH_NOARGS,
        ""},
    {"assign", (PyCFunction) Fastkmeans_assign, METH_VARARGS, ""},
    {NULL} // Sentinel
};
