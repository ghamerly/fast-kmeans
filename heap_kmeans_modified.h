#ifndef HEAP_KMEANS_MODIFIED_H
#define HEAP_KMEANS_MODIFIED_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * OriginalSpaceKmeans is a base class for other algorithms that operate in the
 * same space as the data being clustered (as opposed to kernelized k-means
 * algorithms, which operate in kernel space).
 */

#include "modified_update_triangle_based_kmeans.h"
#include <vector>
#include <numeric>

typedef std::vector<std::pair<double, int> > Heap;

class HeapKmeansModified : public ModifiedUpdateTriangleBasedKmeans {
public:

	HeapKmeansModified() : heaps(NULL), heapBounds(NULL) {
	}

	virtual ~HeapKmeansModified() { free(); }

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
	virtual void free();

	virtual std::string getName() const {
		return "heapmodified";
	}


protected:

	virtual int runThread(int threadId, int maxIterations);

	void update_bounds();

	virtual void calculate_max_upper_bound();

	// Each thread has k heaps, so we have numThreads * k heaps.
	Heap **heaps;

	// The heapBounds essentially accumulate the total distance traveled by
	// each center over the iterations of k-means. This value is used to
	// compare with the heap priority to determine if a point's bounds
	// (lower-upper) are violated (i.e. < 0).
	double *heapBounds;
};

#endif