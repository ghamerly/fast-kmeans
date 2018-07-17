/* AnnulusKmeans wrapper. The comment at the beginning of each function
 * definition demonstrates its usage in Python.
 */

#include "py_annulus.h"

#include "py_assignment.h"
#include "py_dataset.h"


// Annulus instance object


/*
typedef struct {
    PyObject_HEAD
    AnnulusKmeans *instance;

    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} AnnulusObject;
*/


// Object special methods


static int Annulus_init(AnnulusObject *self) {
    // Annulus()

    self->instance = new AnnulusKmeans();
    Py_INCREF(Py_None);
    self->dataset = Py_None;
    Py_INCREF(Py_None);
    self->assignment = Py_None;

    return 0;
}

static void Annulus_dealloc(AnnulusObject *self) {
    delete self->instance;
    Py_DECREF(self->dataset);
    self->dataset = NULL;
    Py_DECREF(self->assignment);
    self->assignment = NULL;

    Py_TYPE(self)->tp_free((PyObject *) self);
}


// Object properties


static PyObject * Annulus_get_centers(AnnulusObject *self, void *closure) {
    // a_annulus.centers

    const Dataset *centers = self->instance->getCenters();
    if (centers == NULL) {
        Py_RETURN_NONE;
    }

    PyObject *args = Py_BuildValue("ii", centers->n, centers->d);
    DatasetObject *centersObj = (DatasetObject *)
        PyObject_CallObject((PyObject *) &DatasetType, args);

    if (PyErr_Occurred()) {
        return NULL;
    }

    // Copy values from centers to preserve constness

    for (int i = 0; i < centers->nd; i++) {
        centersObj->dataset->data[i] = centers->data[i];
    }

    return (PyObject *) centersObj;
}

static PyGetSetDef Annulus_getsetters[] = {
    {
        const_cast<char *>("centers"),
        (getter) Annulus_get_centers,
        NULL, // setter
        const_cast<char *>("The set of centers")
    },
    {NULL} // Sentinel
};


// AnnulusKmeans methods


static PyObject * Annulus_free(AnnulusObject *self) {
    // a_annulus.free()

    self->instance->free();
    Py_RETURN_NONE;
}

static PyObject * Annulus_initialize(AnnulusObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_annulus.initialize(x, k, initial_assignment, num_threads=an_int)

    PyObject *x_orig, *initAssigns_orig;
    unsigned short k;
    int numThreads = 1;

    char *emptyStr = const_cast<char *>("");
    char *kwlist[] = {emptyStr, emptyStr, emptyStr,
        const_cast<char *>("num_threads"), NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!HO!|i", kwlist,
                &DatasetType, &x_orig, &k, &AssignmentType, &initAssigns_orig,
                &numThreads)) {
        return NULL;
    }

    // Reassign dataset
    Py_DECREF(self->dataset);
    Py_INCREF(x_orig);
    self->dataset = x_orig;

    // Reassign assignment
    Py_DECREF(self->assignment);
    Py_INCREF(initAssigns_orig);
    self->assignment = initAssigns_orig;

    // TODO should these just be of type PyObject *?
    // If so, should use member access macros/inline functions instead of direct
    // access
    DatasetObject *x = (DatasetObject *) x_orig;
    AssignmentObject *initAssigns = (AssignmentObject *) initAssigns_orig;

    self->instance->initialize(x->dataset, k, initAssigns->assignment,
            numThreads);

    Py_RETURN_NONE;
}

static PyObject * Annulus_get_name(AnnulusObject *self) {
    // a_annulus.get_name()

    return PyUnicode_FromString(self->instance->getName().c_str());
}


// OriginalSpaceKmeans methods


static PyObject * Annulus_point_point_inner_product(AnnulusObject *self,
        PyObject *args) {
    // a_annulus.point_point_inner_product(x1, x2)

    int x1ndx, x2ndx;
    if (!PyArg_ParseTuple(args, "ii", &x1ndx, &x2ndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointPointInnerProduct(x1ndx, x2ndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Annulus_point_center_inner_product(AnnulusObject *self,
        PyObject *args) {
    // a_annulus.point_center_inner_product(xndx, cndx)

    int xndx;
    unsigned short cndx;
    if (!PyArg_ParseTuple(args, "iH", &xndx, &cndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(xndx, cndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Annulus_center_center_inner_product(AnnulusObject *self,
        PyObject *args) {
    // a_annulus.center_center_inner_product(c1, c2)

    unsigned short c1, c2;
    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(c1, c2);

    return PyFloat_FromDouble(innerProd);
}


// Kmeans methods


static PyObject * Annulus_run(AnnulusObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_annulus.run(max_iterations = 0)

    int maxIterations = 0;
    static char *kwlist[] = {const_cast<char *>("max_iterations"), NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist,
                &maxIterations)) {
        return NULL;
    }

    int numIterations = maxIterations > 0 ?
        self->instance->run(maxIterations) : self->instance->run();

    return PyLong_FromLong(numIterations);
}

static PyObject * Annulus_get_assignment(AnnulusObject *self, PyObject *args) {
    // a_annulus.get_assignment(xndx)

    int xndx;
    if (!PyArg_ParseTuple(args, "i", &xndx)) {
        return NULL;
    }

    int assignment = self->instance->getAssignment(xndx);

    return PyLong_FromLong(assignment);
}

static PyObject * Annulus_verify_assignment(AnnulusObject *self,
        PyObject *args) {
    // a_annulus.verify_assignment(iteration, startndx, endndx)

    int iteration, startNdx, endNdx;
    if (!PyArg_ParseTuple(args, "iii", &iteration, &startNdx, &endNdx)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject * Annulus_get_sse(AnnulusObject *self) {
    // a_annulus.get_sse()

    return PyFloat_FromDouble(self->instance->getSSE());
}

static PyObject * Annulus_point_center_dist_2(AnnulusObject *self,
        PyObject *args) {
    // a_annulus.point_center_dist_2( x1, cndx)

    int x1;
    unsigned short cndx;
    if (!PyArg_ParseTuple(args, "iH", &x1, &cndx)) {
        return NULL;
    }

    double dist2 = self->instance->pointCenterDist2(x1, cndx);

    return PyFloat_FromDouble(dist2);
}

static PyObject * Annulus_center_center_dist_2(AnnulusObject *self,
        PyObject *args) {
    // a_annulus.center_center_dist_2(c1, c2)

    unsigned short c1, c2;
    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double dist2 = self->instance->centerCenterDist2(c1, c2);

    return PyFloat_FromDouble(dist2);
}


// Annulus method definitions


static PyMethodDef Annulus_methods[] = {
    {"run", (PyCFunction) Annulus_run, METH_VARARGS | METH_KEYWORDS,
        "Run threads until convergence or max iters, and returns num iters"},
    {"free", (PyCFunction) Annulus_free, METH_NOARGS, "Free the object's memory"},
    {"initialize", (PyCFunction) Annulus_initialize,
        METH_VARARGS | METH_KEYWORDS,
        "Initialize algorithm at beginning of run() with given data and "
            "initial_assignment, which will be modified to contain final "
            "assignment of clusters"},
    {"point_point_inner_product",
        (PyCFunction) Annulus_point_point_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_inner_product",
        (PyCFunction) Annulus_point_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"center_center_inner_product",
        (PyCFunction) Annulus_center_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_dist_2", (PyCFunction) Annulus_point_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between a point "
            "and center."},
    {"center_center_dist_2", (PyCFunction) Annulus_center_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between two "
            "centers."},
    {"get_assignment", (PyCFunction) Annulus_get_assignment, METH_VARARGS,
        "Get the cluster assignment for the given point index"},
    {"verify_assignment", (PyCFunction) Annulus_verify_assignment, METH_VARARGS,
        "Verify that current assignment is correct, by checking every "
            "point-center distance. For debugging."},
    {"get_sse", (PyCFunction) Annulus_get_sse, METH_NOARGS,
        "Return the sum of squared errors for each cluster"},
    {"get_name", (PyCFunction) Annulus_get_name, METH_NOARGS,
        "Return the algorithm name"},
    {NULL} // Sentinel
};


// Annulus type object


PyTypeObject AnnulusType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fastkmeans.Annulus", // tp_name
    sizeof(AnnulusObject), // tp_basicsize
    0, // tp_itemsize

    (destructor) Annulus_dealloc, // tp_dealloc
    NULL, // tp_print
    NULL, // tp_getattr
    NULL, // tp_setattr
    NULL, // tp_as_sync
    NULL, // tp_repr

    NULL, // tp_as_number
    NULL, // tp_as_sequence
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

    Annulus_methods, // tp_methods
    NULL, // tp_members
    Annulus_getsetters, // tp_getset
    NULL, // tp_base
    NULL, // tp_dict
    NULL, // tp_descr_get
    NULL, // tp_descr_set
    0, // tp_dictoffset
    (initproc) Annulus_init, // tp_init
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
