#include "py_heap.h"
#include "py_dataset.h"
#include "py_assignment.h"

//#include "heap_kmeans.h"


// Heap instance object


/*
typedef struct {
    PyObject_HEAD
    HeapKmeans *instance;

    PyObject *dataset;
    PyObject *assignment;

    // TODO
    // #ifdef COUNT_DISTANCES
    // long long num_distances;
    // #endif
} HeapObject;
*/


// Object special methods


static int Heap_init(HeapObject *self) {
    // Heap()

    self->instance = new HeapKmeans();

    Py_INCREF(Py_None);
    self->dataset = Py_None;
    Py_INCREF(Py_None);
    self->assignment = Py_None;

    return 0;
}

static void Heap_dealloc(HeapObject *self) {
    delete self->instance;

    Py_DECREF(self->dataset);
    self->dataset = NULL;
    Py_DECREF(self->assignment);
    self->assignment = NULL;

    Py_TYPE(self)->tp_free((PyObject *) self);
}


// Object properties


static PyObject * Heap_get_centers(HeapObject *self, void *closure) {
    // a_heap.centers

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

static PyGetSetDef Heap_getsetters[] = {
    // Readonly
    {"centers", (getter) Heap_get_centers, NULL, "The set of centers"},
    {NULL} // Sentinel
};


// HeapKmeans methods


static PyObject * Heap_free(HeapObject *self) {
    // a_heap.free()

    self->instance->free();
    Py_RETURN_NONE;
}

static PyObject * Heap_initialize(HeapObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_heap.initialize(x, k, initial_assignment, num_threads=an_int)

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

static PyObject * Heap_get_name(HeapObject *self) {
    // a_heap.get_name()

    return PyUnicode_FromString(self->instance->getName().c_str());
}


// OriginalSpaceKmeans methods


static PyObject * Heap_point_point_inner_product(HeapObject *self,
        PyObject *args) {
    // a_heap.point_point_inner_product(x1, x2)

    int x1ndx, x2ndx;

    if (!PyArg_ParseTuple(args, "ii", &x1ndx, &x2ndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointPointInnerProduct(x1ndx, x2ndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Heap_point_center_inner_product(HeapObject *self,
        PyObject *args) {
    // a_heap.point_center_inner_product(xndx, cndx)

    int xndx;
    unsigned short cndx;

    if (!PyArg_ParseTuple(args, "iH", &xndx, &cndx)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(xndx, cndx);

    return PyFloat_FromDouble(innerProd);
}

static PyObject * Heap_center_center_inner_product(HeapObject *self,
        PyObject *args) {
    // a_heap.center_center_inner_product(c1, c2)

    unsigned short c1, c2;

    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double innerProd = self->instance->pointCenterInnerProduct(c1, c2);

    return PyFloat_FromDouble(innerProd);
}


// Kmeans methods


static PyObject * Heap_run(HeapObject *self, PyObject *args,
        PyObject *kwargs) {
    // a_heap.run(max_iterations = 0)

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

static PyObject * Heap_get_assignment(HeapObject *self, PyObject *args) {
    // a_heap.get_assignment(xndx)

    int xndx;

    if (!PyArg_ParseTuple(args, "i", &xndx)) {
        return NULL;
    }

    int assignment = self->instance->getAssignment(xndx);

    return PyLong_FromLong(assignment);
}

static PyObject * Heap_verify_assignment(HeapObject *self, PyObject *args) {
    // a_heap.verify_assignment(iteration, startndx, endndx)

    int iteration, startNdx, endNdx;

    if (!PyArg_ParseTuple(args, "iii", &iteration, &startNdx, &endNdx)) {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject * Heap_get_sse(HeapObject *self) {
    // a_heap.get_sse()

    return PyFloat_FromDouble(self->instance->getSSE());
}

static PyObject * Heap_point_center_dist_2(HeapObject *self, PyObject *args) {
    // a_heap.point_center_dist_2( x1, cndx)

    int x1;
    unsigned short cndx;

    if (!PyArg_ParseTuple(args, "iH", &x1, &cndx)) {
        return NULL;
    }

    double dist2 = self->instance->pointCenterDist2(x1, cndx);

    return PyFloat_FromDouble(dist2);
}

static PyObject * Heap_center_center_dist_2(HeapObject *self, PyObject *args) {
    // a_heap.center_center_dist_2(c1, c2)

    unsigned short c1, c2;

    if (!PyArg_ParseTuple(args, "HH", &c1, &c2)) {
        return NULL;
    }

    double dist2 = self->instance->centerCenterDist2(c1, c2);

    return PyFloat_FromDouble(dist2);
}


// Heap method definitions


static PyMethodDef Heap_methods[] = {
    {"run", (PyCFunction) Heap_run, METH_VARARGS | METH_KEYWORDS,
        "Run threads until convergence or max iters, and returns num iters"},
    {"free", (PyCFunction) Heap_free, METH_NOARGS, "Free the object's memory"},
    {"initialize", (PyCFunction) Heap_initialize,
        METH_VARARGS | METH_KEYWORDS,
        "Initialize algorithm at beginning of run() with given data and "
            "initial_assignment, which will be modified to contain final "
            "assignment of clusters"},
    {"point_point_inner_product",
        (PyCFunction) Heap_point_point_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_inner_product",
        (PyCFunction) Heap_point_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"center_center_inner_product",
        (PyCFunction) Heap_center_center_inner_product,
        METH_VARARGS,
        "Compute inner product. Could be standard dot operator, or kernel "
            "function for more exotic applications."},
    {"point_center_dist_2", (PyCFunction) Heap_point_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between a point "
            "and center."},
    {"center_center_dist_2", (PyCFunction) Heap_center_center_dist_2,
        METH_VARARGS,
        "Use the inner products to computer squared distances between two "
            "centers."},
    {"get_assignment", (PyCFunction) Heap_get_assignment, METH_VARARGS,
        "Get the cluster assignment for the given point index"},
    {"verify_assignment", (PyCFunction) Heap_verify_assignment, METH_VARARGS,
        "Verify that current assignment is correct, by checking every "
            "point-center distance. For debugging."},
    {"get_sse", (PyCFunction) Heap_get_sse, METH_NOARGS,
        "Return the sum of squared errors for each cluster"},
    {"get_name", (PyCFunction) Heap_get_name, METH_NOARGS,
        "Return the algorithm name"},
    {NULL} // Sentinel
};


// Heap type object


PyTypeObject HeapType = {
    PyVarObject_HEAD_INIT(NULL, 0)
};

void init_heap_type_fields(void) {
    HeapType.tp_name = "fastkmeans.Heap";
    HeapType.tp_doc = "";
    HeapType.tp_basicsize = sizeof(HeapObject);
    HeapType.tp_itemsize = 0;
    HeapType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    HeapType.tp_new = PyType_GenericNew;
    HeapType.tp_init = (initproc) Heap_init;
    HeapType.tp_dealloc = (destructor) Heap_dealloc;
    HeapType.tp_methods = Heap_methods;
    HeapType.tp_getset = Heap_getsetters;
};
