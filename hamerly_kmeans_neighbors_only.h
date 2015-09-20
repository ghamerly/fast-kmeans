#ifndef HAMERLY_KMEANS_NEIGHBORS_ONLY_H
#define HAMERLY_KMEANS_NEIGHBORS_ONLY_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * HamerlyKmeans implements Hamerly's k-means algorithm that uses one lower
 * bound per point. In this implementation the iteration over neighbors is
 * implemented.
 */

#include "hamerly_kmeans_neighbors.h"

class HamerlyKmeansNeighborsOnly : public HamerlyKmeansNeighbors {
    public:
        HamerlyKmeansNeighborsOnly() {}
        virtual ~HamerlyKmeansNeighborsOnly() { free(); }
        virtual std::string getName() const { return "hamerlyneighborsonly"; }

    protected:
		virtual int move_centers();

		virtual void calculate_lower_bound_update();
};

#endif
