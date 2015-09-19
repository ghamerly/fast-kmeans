#ifndef HEAP_KMEANS_UBARR_NEIGHBORS_H
#define HEAP_KMEANS_UBARR_NEIGHBORS_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * HeapKmeansUBarrNeighbors extends HeapKmeansModifiedUBarr. This algorithm
 * enhances the the previous algorithm by iteration over neighbors, which
 * allows it to eliminate centroids from the innermost loop.
 */

#include "triangle_based_kmeans_neighbors.h"
#include <vector>
#include <numeric>

typedef std::vector<std::pair<double, int> > Heap;

class HeapKmeansUBarrNeighbors : public TriangleBasedKmeansNeighbors {
public:

	HeapKmeansUBarrNeighbors() : heaps(NULL), heapBounds(NULL), maxUBHeap(NULL), ubHeapBounds(NULL) {}

	virtual ~HeapKmeansUBarrNeighbors() {
		free();
	}

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
	virtual void free();

	virtual std::string getName() const {
		return "heapubarrneighbors";
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

	// this will be set of k heaps, one for each centroid
	// each heap will store a pair of upper bound and point index
	// this allows us to find maximum over upper bound
	Heap *maxUBHeap;

	// this array relativizes the upper bound, similarly as the
	// heap key, so that we do not have to update all the n
	// upper bounds, but updating k is enough
	double *ubHeapBounds;
};

#endif