#ifndef ELKAN_KMEANS_MODIFIED_H
#define ELKAN_KMEANS_MODIFIED_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations. This adds to the Elkan's algorithm tighter
 * lower bound udpate.
 */

#include "modified_update_triangle_based_kmeans.h"

class ElkanKmeansModified : public ModifiedUpdateTriangleBasedKmeans {
public:

	ElkanKmeansModified() {};

	virtual ~ElkanKmeansModified() {
		free();
	}

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);

	virtual std::string getName() const {
		return "elkanmodified";
	}

protected:

	// override this as we are calculating each bound explicitly
	virtual void calculate_lower_bound_update();

	virtual int runThread(int threadId, int maxIterations);

	// Update the upper and lower bounds for the range of points given.
	void update_bounds(int startNdx, int endNdx);

};

#endif
