#ifndef TRIANGLE_BASED_KMEANS_NEIGHBORS_H
#define TRIANGLE_BASED_KMEANS_NEIGHBORS_H

/* Authors: Greg Hamerly and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * TriangleBasedKmeansNeighbors is a base class for kmeans algorithm
 * that implement iteration over set of neighbors. It extends the
 * ModifiedUpdateTriangleBasedKmeans and it provides functionality
 * for using iteration over neighobrs.
 */

#include "modified_update_triangle_based_kmeans.h"

class TriangleBasedKmeansNeighbors : public ModifiedUpdateTriangleBasedKmeans {
public:

	TriangleBasedKmeansNeighbors() : neighbours(NULL){}

	virtual ~TriangleBasedKmeansNeighbors() {
		free();
	}

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
	virtual void free();

protected:

	/*
	 * Override the lower bound calculation, so that we do not have to calculate
	 * the neighbour set twice.
	 */
	void calculate_lower_bound_update();

	/*
	 * Here we will store the set of neighbors. It is k array of k arrays, which
	 * contain set of neighbors. For each centroid there is list of neighbors,
	 * that ends by -1, which means end.
	 */
	int** neighbours;
};

#endif
