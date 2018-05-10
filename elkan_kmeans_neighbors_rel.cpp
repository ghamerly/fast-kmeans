/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "elkan_kmeans_neighbors_rel.h"
#include "general_functions.h"
#include <cmath>

void ElkanKmeansNeighborsRel::free() {
    ElkanKmeansNeighbors::free();
    delete [] upperRel;
    delete [] lowerRel;
    upperRel = NULL;
    lowerRel = NULL;
}

/* This function initializes the the upper and lower bound arrays that
 * are used for storing the sum of center movement.
 */
void ElkanKmeansNeighborsRel::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    ElkanKmeansNeighbors::initialize(aX, aK, initialAssignment, aNumThreads);
    upperRel = new double[k];
    std::fill(upperRel, upperRel + k, 0.0);
    lowerRel = new double[k * k];
    std::fill(lowerRel, lowerRel + k*k, 0.0);
}

void ElkanKmeansNeighborsRel::calculate_max_upper_bound(int threadId) {
    ElkanKmeansNeighbors::calculate_max_upper_bound(threadId);
    synchronizeAllThreads();
    // do not forget that the upper bound array is smaller by upperRel
    addVectors(maxUpperBound, upperRel, k);
}

int ElkanKmeansNeighborsRel::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    // here we need to calculate s & the centroid-centroid distances before the first iteration
    // the remaining calls to this method are hidden by move_centers
    update_s(threadId);

    while ((iterations < maxIterations) && !converged) {
        ++iterations;

        // now we need to filter neighbors so that we remain only with those
        // that fulfil the stronger condition that is used for elkan_kmeans
        if (iterations != 1)
            filter_neighbors(threadId);

        synchronizeAllThreads();

        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            bool r = true;
            // the true value of the upper bound
            double upperI = upper[i] + upperRel[closest];

            if (upperI <= s[closest]) {
                continue;
            }

            // iterate only over centroids that can be closest to some x in the dataset
            for (int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr) {
                // now j has the same meaning as before
                int j = (*ptr);
                // here is the true value of the lower bound
                double lowerIJ = lower[i * k + j] - lowerRel[assignment[i] * k + j];
                if (upperI <= lowerIJ) {
                    continue;
                }
                if (upperI <= centerCenterDistDiv2[closest * k + j]) {
                    continue;
                }

                // ELKAN 3(a)
                if (r) {
                    upperI = sqrt(pointCenterDist2(i, closest));
                    // the lower bound has to be stored relative to the movement
                    lower[i * k + closest] = upperI + lowerRel[assignment[i] * k + closest];
                    r = false;
                    if ((upperI <= lowerIJ) || (upperI <= centerCenterDistDiv2[closest * k + j])) {
                        continue;
                    }
                }

                // ELKAN 3(b)
                lowerIJ = sqrt(pointCenterDist2(i, j));
                if (lowerIJ < upperI) {
                    closest = j;
                    upperI = lowerIJ;
                }
                // agan, the lower bound needs to be stored relative to the movement
                lower[i * k + j] = lowerIJ + lowerRel[assignment[i] * k + j];
            }
            // and the upper as well
            upper[i] = upperI - upperRel[assignment[i]];
            if (assignment[i] != closest) {
                // update the upper bound so that it is relative to the new assignment
                upper[i] += upperRel[assignment[i]] - upperRel[closest];
                // do the same for all k lower bounds
                for (int j = 0; j < k; ++j)
                    lower[i * k + j] -= lowerRel[assignment[i] * k + j] - lowerRel[closest * k + j];
                changeAssignment(i, closest, threadId);
            }
        }

        verifyAssignment(iterations, startNdx, endNdx);

        // ELKAN 4, 5, AND 6
        synchronizeAllThreads();
        move_centers(threadId);

        synchronizeAllThreads();
        if (threadId == 0 && !converged)
            update_bounds(startNdx, endNdx);
        synchronizeAllThreads();
    }

    return iterations;
}

void ElkanKmeansNeighborsRel::update_bounds(int startNdx, int endNdx) {
    #ifdef COUNT_DISTANCES
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
            boundsUpdates += ((double) clusterSize[0][i]) * (lowerBoundUpdate[i * k + j]);
        }
    #endif

    // only add the center movement to the upperRel and lower bound update to the lowerRel
    addVectors(upperRel, centerMovement, k);
    addVectors(lowerRel, lowerBoundUpdate, k * k);
}