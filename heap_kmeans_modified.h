#ifndef HEAP_KMEANS_MODIFIED_H
#define HEAP_KMEANS_MODIFIED_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * HeapKmeans keeps k heaps, one for each cluster. The points for each cluster
 * are kept in the respective heaps, ordered by the bound difference
 * (lower-upper), where lower is the same as Hamerly's lower bound. For details
 * on this algorithm, please see the forthcoming book chapter (Partitional
 * Clustering Algorithms, Springer 2014).
 *
 * This class adds to HeapKmeans tigher upper bound update.
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

	// this method is overridden, takes as maximum upper bound the
	// maximum upper bound from the first iteration increased in
	// each iteration by the center movement
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