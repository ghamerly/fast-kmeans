/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "hamerly_kmeans_neighbors.h"
#include "general_functions.h"
#include <cmath>
#include <iostream>

/* Hamerly's algorithm that is a 'simplification' of Elkan's, in that it keeps
 * the following bounds:
 *  - One upper bound per clustered record on the distance between the record
 *    and its closest center. It is always greater than or equal to the true
 *    distance between the record and its closest center. This is the same as in
 *    Elkan's algorithm.
 *  - *One* lower bound per clustered record on the distance between the record
 *    and its *second*-closest center. It is always less than or equal to the
 *    true distance between the record and its second closest center. This is
 *    different information than Elkan's algorithm -- his algorithm keeps k
 *    lower bounds for each record, for a total of (n*k) lower bounds.
 *
 * The basic ideas are:
 *  - when lower(x) <= upper(x), we need to recalculate the closest centers for
 *    the record x, and reset lower(x) and upper(x) to their boundary values
 *  - whenever a center moves
 *      - calculate the distance it moves 'd'
 *      - for each record x assigned to that center, update its upper bound
 *          - upper(x) = upper(x) + d
 *  - after each iteration
 *      - find the center that has moved the most (with distance 'd')
 *      - update the lower bound for all (?) records:
 *          - lower(x) = lower(x) - lower bound update(assignment(x))
 *
 * Parameters: none
 *
 * Return value: the number of iterations performed (always at least 1)
 */
// this version only updates center locations when necessary
int HamerlyKmeansNeighbors::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

	// this must be done before the first iteration and also just after moving the
	// centroids before update - this is different from original hamerly_kmeans
	update_s(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        // compute the inter-center distances, keeping only the closest distances
        synchronizeAllThreads();

        // loop over all records
        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];

            // if upper[i] is less than the greater of these two, then we can
            // ignore record i
            double upper_comparison_bound = std::max(s[closest], lower[i]);

            // first check: if u(x) <= s(c(x)) or u(x) <= lower(x), then ignore
            // x, because its closest center must still be closest
            if (upper[i] <= upper_comparison_bound) {
                continue;
            }

            // otherwise, compute the real distance between this record and its
            // closest center, and update upper
            double u2 = pointCenterDist2(i, closest);
            upper[i] = sqrt(u2);

            // if (u(x) <= s(c(x))) or (u(x) <= lower(x)), then ignore x
            if (upper[i] <= upper_comparison_bound) {
                continue;
            }

            // now update the lower bound by looking at all other centers
            double l2 = std::numeric_limits<double>::max(); // the squared lower bound
            // look over the neighbours of the cluster where the point is assigned
			for(int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr) {
                double dist2 = pointCenterDist2(i, (*ptr));

                if (dist2 < u2) {
                    // another center is closer than the current assignment

                    // change the lower bound to be the current upper bound
                    // (since the current upper bound is the distance to the
                    // now-second-closest known center)
                    l2 = u2;

                    // adjust the upper bound and the current assignment
                    u2 = dist2;
                    closest = (*ptr);
                } else if (dist2 < l2) {
                    // we must reduce the lower bound on the distance to the
                    // *second* closest center to x[i]
                    l2 = dist2;
                }
            }

            // we have been dealing in squared distances; need to convert
            lower[i] = sqrt(l2);

            // if the assignment for i has changed, then adjust the counts and
            // locations of each center's accumulated mass
            if (assignment[i] != closest) {
                upper[i] = sqrt(u2);
                changeAssignment(i, closest, threadId);
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
        // calculate the new center locations
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
