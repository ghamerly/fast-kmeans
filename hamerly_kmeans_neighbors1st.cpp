/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "hamerly_kmeans_neighbors1st.h"
#include "general_functions.h"
#include <cmath>
#include <iostream>

/* This class implements Hamerly's algorithm together with tighter lower bound
 * update and iteration over neighbors (including the first iteration).
 *  - For information about Hamerly's algorithm see hamerly_kmeans.cpp
 *  - For information about tighter lower bound update and neighbors see
 *    modified_update_triangle_based_kmeans.h.
 * The iteration over neighbors is based on the following:
 *  - If the initial assignment is good and tight, we can reuse its results.
 *  - We use the neighbors condition in the first iteration.
 *  - To use it we evaluate the maximum upper bound.
 *  - Therefore we calculate the upper bound separately.
 */
int HamerlyKmeansNeighbors1st::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    // here we need to calculate s & the centroid-centroid distances before the first iteration
    // the remaining calls to this method are hidden by move_centers
	update_s(threadId);

    ++iterations;

    // calculate the distance between each point and its assigned centroid
    // we would have to calculate this anyway
    for(int i = startNdx; i < endNdx; ++i) {
        unsigned short closest = assignment[i];
		upper[i] = sqrt(pointCenterDist2(i, closest));
        // also evaluate the maximum upper bound
		if(upper[i] > maxUpperBound[closest])
			maxUpperBound[closest] = upper[i];
	}

	// now get the neighbours, but we cannot use the superclass as centroids have
    // not moved yet, therefore there is no centerMovement
	for(int C = 0; C < k; ++C) {
		double boundOnOtherDistance = maxUpperBound[C] + s[C];
		int neighboursPos = 0;
		for (int c = 0; c < k; ++c)
		{
			if(c != C && boundOnOtherDistance >= centerCenterDistDiv2[C * k + c])
				neighbours[C][neighboursPos++] = c;
		}
		neighbours[C][neighboursPos] = -1;
	}

    for(int i = startNdx; i < endNdx; ++i) {
        unsigned short closest = assignment[i];
        // check that we can skip the distance calculations
		if(s[closest] >= upper[i]) {
            // if so, initialize the lower bound
			lower[i] = 2*s[closest] - upper[i];
			continue;
		}

        // we cannot tighten the upper bound - it is already tight, skip it therefore

		double u2 = upper[i] * upper[i];
        double l2 = std::numeric_limits<double>::max(); // the squared lower bound
        // now iterate over the neighbors
        for(int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr) {
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

    if (! converged) {
        update_bounds(startNdx, endNdx);
    }

    synchronizeAllThreads();

	// end of the first iteration

    // ... now do everything in the same way as we did before
    // therefore call the superclass
    // note that this implementation costs us unnecessary k*k-k distance calculations
    // on update_s() function - this k*(k-1) distances can be eliminated by the change
    return HamerlyKmeansNeighbors::runThread(threadId, maxIterations - 1) + 1;
}
