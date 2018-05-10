/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "hamerly_kmeans_neighbors.h"
#include "general_functions.h"
#include <cmath>

/* This class implements Hamerly's algorithm together with tighter lower bound
 * update and iteration over neighbors.
 *  - For information about Hamerly's algorithm see hamerly_kmeans.cpp
 *  - For information about tighter lower bound update and neighbors see
 *    modified_update_triangle_based_kmeans.h.
 */
int HamerlyKmeansNeighbors::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    // here we need to calculate s & the centroid-centroid distances before the first iteration
    // the remaining calls to this method are hidden by move_centers
    update_s(threadId);
    synchronizeAllThreads();

    while ((iterations < maxIterations) && !converged) {
        ++iterations;

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
            // iterate only over centroids that can be closest or second closest to some x in the dataset
            for (int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr) {
                double dist2 = pointCenterDist2(i, (*ptr));

                if (dist2 < u2) {
                    l2 = u2;
                    u2 = dist2;
                    closest = (*ptr);
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
        move_centers(threadId);

        synchronizeAllThreads();

        if (!converged) {
            update_bounds(startNdx, endNdx);
        }

        synchronizeAllThreads();
    }

    return iterations;
}
