#ifndef HEAP_KMEANS_UBARR_NEIGHBORS_H
#define HEAP_KMEANS_UBARR_NEIGHBORS_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * HeapKmeansUBarrNeighbors extends HeapKmeansModifiedUBarr. This algorithm
 * enhances the the previous algorithm by iteration over neighbors, which
 * allows it to eliminate centroids from the innermost loop. Everything is
 * exactly same as in the parent class.
 */

#include "heap_kmeans_ubarr.h"
#include <vector>

typedef std::vector<std::pair<double, int> > Heap;

class HeapKmeansUBarrNeighbors : public HeapKmeansUBarr {
public:

	HeapKmeansUBarrNeighbors() {}

	virtual ~HeapKmeansUBarrNeighbors() {
		free();
	}

	virtual std::string getName() const {
		return "heapubarrneighbors";
	}


protected:

	virtual int runThread(int threadId, int maxIterations);
};

#endif