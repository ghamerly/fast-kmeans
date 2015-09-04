#ifndef HAMERLY_KMEANS_NEIGHBORS_H
#define HAMERLY_KMEANS_NEIGHBORS_H

/* Authors: Greg Hamerly, Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * HamerlyKmeans implements Hamerly's k-means algorithm that uses one lower
 * bound per point.
 *
 * This adds to the Hamerly's algorith tighter lower bound update and
 * iteration over neighbours.
 */

#include "triangle_based_kmeans_neighbors.h"

class HamerlyKmeansNeighbors : public TriangleBasedKmeansNeighbors {
    public:
        HamerlyKmeansNeighbors() { numLowerBounds = 1; }
        virtual ~HamerlyKmeansNeighbors() { free(); }
        virtual std::string getName() const { return "hamerlyneighbors"; }

    protected:
        // Update the upper and lower bounds for the given range of points.
        void update_bounds(int startNdx, int endNdx);

        virtual int runThread(int threadId, int maxIterations);
};

#endif
