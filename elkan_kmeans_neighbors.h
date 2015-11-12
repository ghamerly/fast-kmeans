#ifndef ELKAN_KMEANS_NEIGHBORS_H
#define ELKAN_KMEANS_NEIGHBORS_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * Elkan's k-means algorithm that uses k lower bounds per point to prune
 * distance calculations. This adds to the Elkan's algorithm tighter
 * lower bound udpate and iteration over neighbors. Note that for Elkan's
 * algorithm this condition is more strong than for other algorithms.
 */

#include "elkan_kmeans_modified.h"

class ElkanKmeansNeighbors : public ElkanKmeansModified {
public:

	ElkanKmeansNeighbors() {};

	virtual ~ElkanKmeansNeighbors() { free(); }

	virtual std::string getName() const {
		return "elkanneighbors";
	}

protected:

	/*
	 * This method filter neighbors so that they contain only points
	 * that fulfil the stronger condition for elkan kmeans, i.e.
	 * m(cj) > 1/2 \| ci - cj \|.
	 */
	void filter_neighbors(int threadId);

	virtual int runThread(int threadId, int maxIterations);

};

#endif
