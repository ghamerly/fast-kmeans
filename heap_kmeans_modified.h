#ifndef HEAP_KMEANS_MODIFIED_H
#define HEAP_KMEANS_MODIFIED_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * This class is extension to the HeapKmeans. It implement the tigher upper bound update
 * that uses only estimate of m(c_i). To get more information about heap kmeans see
 * heap_kmeans.cpp/h. To get information about the tighter update see
 * modified_update_triangle_based_kmeans.h. The most of the code is inherited from the
 * parent class or copied from the HeapKmeans.h.
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
	virtual void calculate_max_upper_bound(int threadId);

	Heap **heaps;

	double *heapBounds;
};

#endif