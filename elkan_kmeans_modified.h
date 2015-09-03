#ifndef ELKAN_KMEANS_MODIFIED_H
#define ELKAN_KMEANS_MODIFIED_H

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
class ElkanKmeansModified : public ModifiedUpdateTriangleBasedKmeans {
public:

	ElkanKmeansModified() {};

	virtual ~ElkanKmeansModified() {
		free();
	}
//	virtual void free();
	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);

	virtual std::string getName() const {
		return "elkanmodified";
	}

protected:

	virtual void calculate_lower_bound_update();

	virtual int runThread(int threadId, int maxIterations);

	// Update the upper and lower bounds for the range of points given.
	void update_bounds(int startNdx, int endNdx);

};

#endif
