#include "py_compare.h"
#include "py_dataset.h"
#include "py_assignment.h"

//#include "compare_kmeans.h"


// Compare instance object


/*
typedef struct {
    PyObject_HEAD
    CompareKmeans *instance;

    // TODO get_centers returns a constant Dataset, so if Dataset object is not
    // constant, this should convert it to const e.g. tuple of tuples
    // PyObject *centers; // Maybe a list

    // PyObject *sum_new_centers; // list of Dataset *

    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} CompareObject;
*/


// Object special methods


static int Compare_init(CompareObject *self) {
    // Compare()

    self->instance = new CompareKmeans();

    Py_INCREF(Py_None);
    self->dataset = Py_None;
    Py_INCREF(Py_None);
    self->assignment = Py_None;

    return 0;
}

static void Compare_dealloc(CompareObject *self) {
    delete self->instance;

    Py_DECREF(self->dataset);
    self->dataset = NULL;
    Py_DECREF(self->assignment);
    self->assignment = NULL;

    Py_TYPE(self)->tp_free((PyObject *) self);
}


// Object properties


static PyObject * Compare_get_centers(CompareObject *self, void *closure) {
    // a_compare.centers

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

static PyGetSetDef Compare_getsetters[] = {
    // Readonly
    {"centers", (getter) Compare_get_centers, NULL, "The set of centers"},
    {NULL} // Sentinel
};


// CompareKmeans methods


static PyObject * Compare_free(CompareObject *self) {
    // Compare.free(self)

    self->instance->free();
    Py_RETURN_NONE;
}

static PyObject * Compare_initialize(CompareObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_compare.initialize(x, k, initial_assignment, num_threads)

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

static PyObject * Compare_get_name(CompareObject *self) {
    // a_compare.get_name()

    return PyUnicode_FromString(self->instance->getName().c_str());
}


// OriginalSpaceKmeans methods


static PyObject * Compare_point_point_inner_product(CompareObject *self,
        PyObject *args) {
    // a_compare.point_point_inner_product(x1, x2)

    int x1ndx, x2ndx;

    if (!PyArg_ParseTuple(args, "ii", &x1ndx, &x2ndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointPointInnerProduct(x1ndx, x2ndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Compare_point_center_inner_product(CompareObject *self,
        PyObject *args) {
    // a_compare.point_center_inner_product(xndx, cndx)

    int xndx;
    unsigned short cndx;

    if (!PyArg_ParseTuple(args, "iH", &xndx, &cndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(xndx, cndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Compare_center_center_inner_product(CompareObject *self,
        PyObject *args) {
    // a_compare.center_center_inner_product(c1, c2)

    unsigned short c1, c2;

    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(c1, c2);

    return PyFloat_FromDouble(innerProd);
}


// Kmeans methods


static PyObject * Compare_run(CompareObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_compare.run(max_iterations = 0)

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

static PyObject * Compare_get_assignment(CompareObject *self, PyObject *args) {
    // a_compare.get_assignment(xndx)

    int xndx;

    if (!PyArg_ParseTuple(args, "i", &xndx)) {
        return NULL;
    }

    int assignment = self->instance->getAssignment(xndx);

    return PyLong_FromLong(assignment);
}

static PyObject * Compare_verify_assignment(CompareObject *self, PyObject *args) {
    // a_compare.verify_assignment(iteration, startndx, endndx)

    int iteration, startNdx, endNdx;

    if (!PyArg_ParseTuple(args, "iii", &iteration, &startNdx, &endNdx)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject * Compare_get_sse(CompareObject *self) {
    // a_compare.get_sse()

    return PyFloat_FromDouble(self->instance->getSSE());
}

static PyObject * Compare_point_center_dist_2(CompareObject *self, PyObject *args) {
    // a_compare.point_center_dist_2( x1, cndx)

    int x1;
    unsigned short cndx;

    if (!PyArg_ParseTuple(args, "iH", &x1, &cndx)) {
        return NULL;
    }

    double dist2 = self->instance->pointCenterDist2(x1, cndx);

    return PyFloat_FromDouble(dist2);
}

static PyObject * Compare_center_center_dist_2(CompareObject *self, PyObject *args) {
    // a_compare.center_center_dist_2(c1, c2)

    unsigned short c1, c2;

    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double dist2 = self->instance->centerCenterDist2(c1, c2);

    return PyFloat_FromDouble(dist2);
}


// Compare method definitions


static PyMethodDef Compare_methods[] = {
    {"run", (PyCFunction) Compare_run, METH_VARARGS | METH_KEYWORDS,
        "Run threads until convergence or max iters, and returns num iters"},
    {"free", (PyCFunction) Compare_free, METH_NOARGS, "Free the object's memory"},
    {"initialize", (PyCFunction) Compare_initialize,
        METH_VARARGS | METH_KEYWORDS,
        "Initialize algorithm at beginning of run() with given data and "
            "initial_assignment, which will be modified to contain final "
            "assignment of clusters"},
    {"point_point_inner_product",
        (PyCFunction) Compare_point_point_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_inner_product",
        (PyCFunction) Compare_point_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"center_center_inner_product",
        (PyCFunction) Compare_center_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_dist_2", (PyCFunction) Compare_point_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between a point "
            "and center."},
    {"center_center_dist_2", (PyCFunction) Compare_center_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between two "
            "centers."},
    {"get_assignment", (PyCFunction) Compare_get_assignment, METH_VARARGS,
        "Get the cluster assignment for the given point index"},
    {"verify_assignment", (PyCFunction) Compare_verify_assignment, METH_VARARGS,
        "Verify that current assignment is correct, by checking every "
            "point-center distance. For debugging."},
    {"get_sse", (PyCFunction) Compare_get_sse, METH_NOARGS,
        "Return the sum of squared errors for each cluster"},
    {"get_name", (PyCFunction) Compare_get_name, METH_NOARGS,
        "Return the algorithm name"},
    {NULL} // Sentinel
};


// Compare type object


PyTypeObject CompareType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

void init_compare_type_fields(void) {
    CompareType.tp_name = "fastkmeans.Compare";
    CompareType.tp_doc = "";
    CompareType.tp_basicsize = sizeof(CompareObject);
    CompareType.tp_itemsize = 0;
    CompareType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    CompareType.tp_new = PyType_GenericNew;
    CompareType.tp_init = (initproc) Compare_init;
    CompareType.tp_dealloc = (destructor) Compare_dealloc;
    CompareType.tp_methods = Compare_methods;
    CompareType.tp_getset = Compare_getsetters;
};
