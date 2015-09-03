#ifndef TRIANGLE_BASED_KMEANS_NEIGHBORS_H
#define TRIANGLE_BASED_KMEANS_NEIGHBORS_H

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

/* Cluster with the cluster centers living in the original space (with the
 * data). This is as opposed to a kernelized version of k-means, where the
 * center points might not be explicitly represented. This is also an abstract
 * class.
 */
class TriangleBasedKmeansNeighbors : public ModifiedUpdateTriangleBasedKmeans {
public:

	TriangleBasedKmeansNeighbors() : neighbours(NULL){}

	virtual ~TriangleBasedKmeansNeighbors() {
		free();
	}

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
	virtual void free();

protected:

	void calculate_lower_bound_update();

	int** neighbours;
};

#endif
