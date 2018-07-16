/* HamerlyKmeans wrapper. The comment at the beginning of each function
 * definition demonstrates its usage in Python.
 */

#include "py_hamerly.h"

#include "py_assignment.h"
#include "py_dataset.h"


// Hamerly instance object


/*
typedef struct {
    PyObject_HEAD
    HamerlyKmeans *instance;

    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} HamerlyObject;
*/


// Object special methods


static int Hamerly_init(HamerlyObject *self) {
    // Hamerly()

    self->instance = new HamerlyKmeans();
    Py_INCREF(Py_None);
    self->dataset = Py_None;
    Py_INCREF(Py_None);
    self->assignment = Py_None;

    return 0;
}

static void Hamerly_dealloc(HamerlyObject *self) {
    delete self->instance;
    Py_DECREF(self->dataset);
    self->dataset = NULL;
    Py_DECREF(self->assignment);
    self->assignment = NULL;

    Py_TYPE(self)->tp_free((PyObject *) self);
}


// Object properties


static PyObject * Hamerly_get_centers(HamerlyObject *self, void *closure) {
    // a_hamerly.centers

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

static PyGetSetDef Hamerly_getsetters[] = {
    {
        const_cast<char *>("centers"),
        (getter) Hamerly_get_centers,
        NULL, // setter
        const_cast<char *>("The set of centers")
    },
    {NULL} // Sentinel
};


// HamerlyKmeans methods


static PyObject * Hamerly_get_name(HamerlyObject *self) {
    // a_hamerly.get_name()

    return PyUnicode_FromString(self->instance->getName().c_str());
}


// TriangleInequalitybaseKmeans methods


static PyObject * Hamerly_free(HamerlyObject *self) {
    // a_hamerly.free()

    self->instance->free();
    Py_RETURN_NONE;
}

static PyObject * Hamerly_initialize(HamerlyObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_hamerly.initialize(x, k, initial_assignment, num_threads=an_int)

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

    // Reassign datset
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


// OriginalSpaceKmeans methods


static PyObject * Hamerly_point_point_inner_product(HamerlyObject *self,
        PyObject *args) {
    // a_hamerly.point_point_inner_product(x1, x2)

    int x1ndx, x2ndx;
    if (!PyArg_ParseTuple(args, "ii", &x1ndx, &x2ndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointPointInnerProduct(x1ndx, x2ndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Hamerly_point_center_inner_product(HamerlyObject *self,
        PyObject *args) {
    // a_hamerly.point_center_inner_product(xndx, cndx)

    int xndx;
    unsigned short cndx;
    if (!PyArg_ParseTuple(args, "iH", &xndx, &cndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(xndx, cndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Hamerly_center_center_inner_product(HamerlyObject *self,
        PyObject *args) {
    // a_hamerly.center_center_inner_product(c1, c2)

    unsigned short c1, c2;
    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(c1, c2);

    return PyFloat_FromDouble(innerProd);
}


// Kmeans methods


static PyObject * Hamerly_run(HamerlyObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_hamerly.run(max_iterations = 0)

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

static PyObject * Hamerly_get_assignment(HamerlyObject *self, PyObject *args) {
    // a_hamerly.get_assignment(xndx)

    int xndx;
    if (!PyArg_ParseTuple(args, "i", &xndx)) {
        return NULL;
    }

    int assignment = self->instance->getAssignment(xndx);

    return PyLong_FromLong(assignment);
}

static PyObject * Hamerly_verify_assignment(HamerlyObject *self, PyObject *args) {
    // a_hamerly.verify_assignment(iteration, startndx, endndx)

    int iteration, startNdx, endNdx;
    if (!PyArg_ParseTuple(args, "iii", &iteration, &startNdx, &endNdx)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject * Hamerly_get_sse(HamerlyObject *self) {
    // a_hamerly.get_sse()

    return PyFloat_FromDouble(self->instance->getSSE());
}

static PyObject * Hamerly_point_center_dist_2(HamerlyObject *self, PyObject *args) {
    // a_hamerly.point_center_dist_2( x1, cndx)

    int x1;
    unsigned short cndx;
    if (!PyArg_ParseTuple(args, "iH", &x1, &cndx)) {
        return NULL;
    }

    double dist2 = self->instance->pointCenterDist2(x1, cndx);

    return PyFloat_FromDouble(dist2);
}

static PyObject * Hamerly_center_center_dist_2(HamerlyObject *self, PyObject *args) {
    // a_hamerly.center_center_dist_2(c1, c2)

    unsigned short c1, c2;
    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double dist2 = self->instance->centerCenterDist2(c1, c2);

    return PyFloat_FromDouble(dist2);
}


// Hamerly method definitions


static PyMethodDef Hamerly_methods[] = {
    {"run", (PyCFunction) Hamerly_run, METH_VARARGS | METH_KEYWORDS,
        "Run threads until convergence or max iters, and returns num iters"},
    {"free", (PyCFunction) Hamerly_free, METH_NOARGS, "Free the object's memory"},
    {"initialize", (PyCFunction) Hamerly_initialize,
        METH_VARARGS | METH_KEYWORDS,
        "Initialize algorithm at beginning of run() with given data and "
            "initial_assignment, which will be modified to contain final "
            "assignment of clusters"},
    {"point_point_inner_product",
        (PyCFunction) Hamerly_point_point_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_inner_product",
        (PyCFunction) Hamerly_point_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"center_center_inner_product",
        (PyCFunction) Hamerly_center_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_dist_2", (PyCFunction) Hamerly_point_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between a point "
            "and center."},
    {"center_center_dist_2", (PyCFunction) Hamerly_center_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between two "
            "centers."},
    {"get_assignment", (PyCFunction) Hamerly_get_assignment, METH_VARARGS,
        "Get the cluster assignment for the given point index"},
    {"verify_assignment", (PyCFunction) Hamerly_verify_assignment, METH_VARARGS,
        "Verify that current assignment is correct, by checking every "
            "point-center distance. For debugging."},
    {"get_sse", (PyCFunction) Hamerly_get_sse, METH_NOARGS,
        "Return the sum of squared errors for each cluster"},
    {"get_name", (PyCFunction) Hamerly_get_name, METH_NOARGS,
        "Return the algorithm name"},
    {NULL} // Sentinel
};


// Hamerly type object


PyTypeObject HamerlyType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "fastkmeans.Hamerly", // tp_name
    sizeof(HamerlyObject), // tp_basicsize
    0, // tp_itemsize

    (destructor) Hamerly_dealloc, // tp_dealloc
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

    Hamerly_methods, // tp_methods
    NULL, // tp_members
    Hamerly_getsetters, // tp_getset
    NULL, // tp_base
    NULL, // tp_dict
    NULL, // tp_descr_get
    NULL, // tp_descr_set
    0, // tp_dictoffset
    (initproc) Hamerly_init, // tp_init
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
