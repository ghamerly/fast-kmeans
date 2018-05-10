#ifndef HAMERLY_KMEANS_NEIGHBORS_ONLY_H
#define HAMERLY_KMEANS_NEIGHBORS_ONLY_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * HamerlyKmeans implements Hamerly's k-means algorithm that uses one lower
 * bound per point.
 *
 * In this implementation the iteration over neighbors is implemented.
 * - For information about neighbors see
 *    modified_update_triangle_based_kmeans.h
 *   note the the tighter lower bound update is not used.
 */

#include "hamerly_kmeans_neighbors.h"

class HamerlyKmeansNeighborsOnly : public HamerlyKmeansNeighbors {
    public:
        HamerlyKmeansNeighborsOnly() {}
        virtual ~HamerlyKmeansNeighborsOnly() { free(); }
        virtual std::string getName() const { return "hamerlyneighborsonly"; }

    protected:
		// we have to override this method in order to skip the tighter lower
		// bound update calculation
		virtual void move_centers(int threadId);

		// here we need to consider only the centroid movement
		// therefore we get the default update given by the triangle inequality
		virtual void calculate_lower_bound_update(int threadId);
};

#endif
