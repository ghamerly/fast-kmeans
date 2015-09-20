#ifndef ELKAN_KMEANS_NEIGHBORS_REL_H
#define ELKAN_KMEANS_NEIGHBORS_REL_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations. This adds to the Elkan's algorithm tighter
 * lower bound udpate, iteration over neighbors and also stores the upper/lower
 * bounds relative to the center movements.
 */

#include "elkan_kmeans_neighbors.h"

class ElkanKmeansNeighborsRel : public ElkanKmeansNeighbors {
public:

	ElkanKmeansNeighborsRel() : upperRel(NULL), lowerRel(NULL) {};

	virtual ~ElkanKmeansNeighborsRel() { free(); }

	virtual std::string getName() const {
		return "elkanneighborsrel";
	}

	virtual void free();
	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);

protected:

	virtual int runThread(int threadId, int maxIterations);

	// now we have effectively to update k bounds, instead of n
	void update_bounds(int startNdx, int endNdx);

	// the maximum upper bound needs to be updated by the relative movement
	virtual void calculate_max_upper_bound();

	// in those two arrays we will store how much is changed the upper/lower
	// bound so that we do not need to update them, but only the accumulated
	// center movement
	// the upper bound is relative to the movement of the assigned centroid
	double * upperRel;
	// and each lower bound is relative to the sum of all lower bound updates
	// that were calculated for this bound
	double * lowerRel;

};

#endif
