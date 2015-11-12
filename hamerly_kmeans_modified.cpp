/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "hamerly_kmeans_modified.h"
#include "general_functions.h"
#include <cmath>
#include <iostream>

/* This class is extension to Hamerly's algorithm with tighter lower bound update.
 *  - For information about Hamerly's algorithm see hamerly_kmeans.cpp
 *  - For information about tighter lower bound update see
 *    modified_update_triangle_based_kmeans.h.
 */
int HamerlyKmeansModified::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    // here we need to calculate s & the centroid-centroid distances before the first iteration
    // the remaining calls to this method are hidden by move_centers
	update_s(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        synchronizeAllThreads();

        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];

            double upper_comparison_bound = std::max(s[closest], lower[i]);

            if (upper[i] <= upper_comparison_bound) {
                continue;
            }

            double u2 = pointCenterDist2(i, closest);
            upper[i] = sqrt(u2);

            if (upper[i] <= upper_comparison_bound) {
                continue;
            }

            double l2 = std::numeric_limits<double>::max(); // the squared lower bound
            for (int j = 0; j < k; ++j) {
                if (j == closest) { continue; }
                double dist2 = pointCenterDist2(i, j);

                if (dist2 < u2) {
                    l2 = u2;
                    u2 = dist2;
                    closest = j;
                } else if (dist2 < l2) {
                    l2 = dist2;
                }
            }

            lower[i] = sqrt(l2);

            if (assignment[i] != closest) {
                upper[i] = sqrt(u2);
                changeAssignment(i, closest, threadId);
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        synchronizeAllThreads();
        if (threadId == 0) {
            int furthestMovingCenter = move_centers();
            converged = (0.0 == centerMovement[furthestMovingCenter]);
        }

        synchronizeAllThreads();

        if (! converged) {
            update_bounds(startNdx, endNdx);
        }

        synchronizeAllThreads();
    }

    return iterations;
}


void HamerlyKmeansModified::update_bounds(int startNdx, int endNdx) {
#ifdef COUNT_DISTANCES
	for(int i = 0; i < k; ++i)
		boundsUpdates += ((double) clusterSize[0][i]) * (lowerBoundUpdate[i]);
#endif

    // update upper/lower bounds
    for (int i = startNdx; i < endNdx; ++i) {
        // the upper bound increases by the amount that its center moved
        upper[i] += centerMovement[assignment[i]];

        // The lower bound decreases by the update that was calculated by
        // the superclass
        lower[i] -= lowerBoundUpdate[assignment[i]];
    }
}
