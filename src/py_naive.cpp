#include "py_naive.h"
#include "py_dataset.h"
#include "py_assignment.h"

//#include "naive_kmeans.h"


// Naive instance object


/*
typedef struct {
    PyObject_HEAD
    NaiveKmeans *instance;

    // TODO get_centers returns a constant Dataset, so if Dataset object is not
    // constant, this should convert it to const e.g. tuple of tuples
    // PyObject *centers; // Maybe a list

    // PyObject *sum_new_centers; // list of Dataset *

    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} NaiveObject;
*/


// Object special methods


static int Naive_init(NaiveObject *self) {
    // Naive()

    self->instance = new NaiveKmeans();

    Py_INCREF(Py_None);
    self->dataset = Py_None;
    Py_INCREF(Py_None);
    self->assignment = Py_None;

    return 0;
}

static void Naive_dealloc(NaiveObject *self) {
    delete self->instance;

    Py_DECREF(self->dataset);
    self->dataset = NULL;
    Py_DECREF(self->assignment);
    self->assignment = NULL;

    Py_TYPE(self)->tp_free((PyObject *) self);
}


// Object properties


static PyObject * Naive_get_centers(NaiveObject *self, void *closure) {
    // a_naive.centers

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

    // Dataset *newCenters = new Dataset(centers->n, centers->d);
    for (int i = 0; i < centers->nd; i++) {
        //newCenters->data[i] = centers->data[i];
        centersObj->dataset->data[i] = centers->data[i];
    }

    //centersObj->dataset = newCenters;

    return (PyObject *) centersObj;
}

static PyGetSetDef Naive_getsetters[] = {
    // Readonly
    {"centers", (getter) Naive_get_centers, NULL, "The set of centers"},
    {NULL} // Sentinel
};


// NaiveKmeans methods


static PyObject * Naive_get_name(NaiveObject *self) {
    // a_naive.get_name()

    return PyUnicode_FromString(self->instance->getName().c_str());
}


// OriginalSpaceKmeans methods


static PyObject * Naive_free(NaiveObject *self) {
    // Naive.free(self)

    self->instance->free();
    Py_RETURN_NONE;
}

static PyObject * Naive_initialize(NaiveObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_naive.initialize(x, k, initial_assignment, num_threads)

    PyObject *x_orig, *initAssigns_orig;
    unsigned short k;
    int numThreads = 1;

    char *kwlist[] = {"", "", "", "num_threads", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!HO!|i", kwlist,
                &DatasetType, &x_orig, &k, &AssignmentType, &initAssigns_orig,
                &numThreads)) {
        return NULL;
    }

    Py_DECREF(self->dataset);
    Py_INCREF(x_orig);
    self->dataset = x_orig;

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

static PyObject * Naive_point_point_inner_product(NaiveObject *self,
        PyObject *args) {
    // a_naive.point_point_inner_product(x1, x2)

    int x1ndx, x2ndx;

    if (!PyArg_ParseTuple(args, "ii", &x1ndx, &x2ndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointPointInnerProduct(x1ndx, x2ndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Naive_point_center_inner_product(NaiveObject *self,
        PyObject *args) {
    // a_naive.point_center_inner_product(xndx, cndx)

    int xndx;
    unsigned short cndx;

    if (!PyArg_ParseTuple(args, "iH", &xndx, &cndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(xndx, cndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Naive_center_center_inner_product(NaiveObject *self,
        PyObject *args) {
    // a_naive.center_center_inner_product(c1, c2)

    unsigned short c1, c2;

    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(c1, c2);

    return PyFloat_FromDouble(innerProd);
}


// Kmeans methods


static PyObject * Naive_run(NaiveObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_naive.run(max_iterations = 0)

    static char *kwlist[] = {"max_iterations"};
    int maxIterations = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist,
                &maxIterations)) {
        return NULL;
    }
    
    int numIterations = maxIterations > 0 ?
        self->instance->run(maxIterations) : self->instance->run();

    return PyLong_FromLong(numIterations);
}

static PyObject * Naive_get_assignment(NaiveObject *self, PyObject *args) {
    // a_naive.get_assignment(xndx)

    int xndx;

    if (!PyArg_ParseTuple(args, "i", &xndx)) {
        return NULL;
    }

    int assignment = self->instance->getAssignment(xndx);

    return PyLong_FromLong(assignment);
}

static PyObject * Naive_verify_assignment(NaiveObject *self, PyObject *args) {
    // a_naive.verify_assignment(iteration, startndx, endndx)

    int iteration, startNdx, endNdx;

    if (!PyArg_ParseTuple(args, "iii", &iteration, &startNdx, &endNdx)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject * Naive_get_sse(NaiveObject *self) {
    // a_naive.get_sse()

    return PyFloat_FromDouble(self->instance->getSSE());
}

static PyObject * Naive_point_center_dist_2(NaiveObject *self, PyObject *args) {
    // a_naive.point_center_dist_2( x1, cndx)

    int x1;
    unsigned short cndx;

    if (!PyArg_ParseTuple(args, "iH", &x1, &cndx)) {
        return NULL;
    }

    double dist2 = self->instance->pointCenterDist2(x1, cndx);

    return PyFloat_FromDouble(dist2);
}

static PyObject * Naive_center_center_dist_2(NaiveObject *self, PyObject *args) {
    // a_naive.center_center_dist_2(c1, c2)

    unsigned short c1, c2;

    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double dist2 = self->instance->centerCenterDist2(c1, c2);

    return PyFloat_FromDouble(dist2);
}


// Naive method definitions


static PyMethodDef Naive_methods[] = {
    {"run", (PyCFunction) Naive_run, METH_VARARGS | METH_KEYWORDS,
        "Run threads until convergence or max iters, and returns num iters"},
    {"free", (PyCFunction) Naive_free, METH_NOARGS, "Free the object's memory"},
    {"initialize", (PyCFunction) Naive_initialize,
        METH_VARARGS | METH_KEYWORDS,
        "Initialize algorithm at beginning of run() with given data and "
            "initial_assignment, which will be modified to contain final "
            "assignment of clusters"},
    {"point_point_inner_product",
        (PyCFunction) Naive_point_point_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_inner_product",
        (PyCFunction) Naive_point_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"center_center_inner_product",
        (PyCFunction) Naive_center_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_dist_2", (PyCFunction) Naive_point_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between a point "
            "and center."},
    {"center_center_dist_2", (PyCFunction) Naive_center_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between two "
            "centers."},
    {"get_assignment", (PyCFunction) Naive_get_assignment, METH_VARARGS,
        "Get the cluster assignment for the given point index"},
    {"verify_assignment", (PyCFunction) Naive_verify_assignment, METH_VARARGS,
        "Verify that current assignment is correct, by checking every "
            "point-center distance. For debugging."},
    {"get_sse", (PyCFunction) Naive_get_sse, METH_NOARGS,
        "Return the sum of squared errors for each cluster"},
    {"get_name", (PyCFunction) Naive_get_name, METH_NOARGS,
        "Return the algorithm name"},
    {NULL} // Sentinel
};


// Naive type object


PyTypeObject NaiveType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

void init_naive_type_fields(void) {
    NaiveType.tp_name = "fastkmeans.Naive";
    NaiveType.tp_doc = "";
    NaiveType.tp_basicsize = sizeof(NaiveObject);
    NaiveType.tp_itemsize = 0;
    NaiveType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    NaiveType.tp_new = PyType_GenericNew;
    NaiveType.tp_init = (initproc) Naive_init;
    NaiveType.tp_dealloc = (destructor) Naive_dealloc;
    NaiveType.tp_methods = Naive_methods;
    NaiveType.tp_getset = Naive_getsetters;
};
