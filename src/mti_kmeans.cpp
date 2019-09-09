#include "mti_kmeans.h"
#include "general_functions.h"
#include <cmath>

void MTIKmeans::update_center_dists(int threadId) {
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1) {
        if (c1 % numThreads == threadId) {
            s[c1] = std::numeric_limits<double>::max();

            for (int c2 = 0; c2 < k; ++c2) {
                // we do not need to consider the case when c1 == c2 as centerCenterDistDiv2[c1*k+c1]
                // is equal to zero from initialization, also this distance should not be used for s[c1]
                if (c1 != c2) {
                    // divide by 2 here since we always use the inter-center
                    // distances divided by 2
                    centerCenterDistDiv2[c1 * k + c2] = sqrt(centerCenterDist2(c1, c2)) / 2.0;

                    if (centerCenterDistDiv2[c1 * k + c2] < s[c1]) {
                        s[c1] = centerCenterDistDiv2[c1 * k + c2];
                    }
                }
            }
        }
    }
}

int MTIKmeans::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        update_center_dists(threadId);
        synchronizeAllThreads();

        for (int i = startNdx; i < endNdx; ++i) {
            unsigned short closest = assignment[i];
            bool r = true;

            // MTI clause I
            if (upper[i] <= s[closest]) {
                continue;
            }

            for (int j = 0; j < k; ++j) {

                // MTI clause II
                if (j == closest ||
                        upper[i] <= centerCenterDistDiv2[closest * k + j]) {
                    continue;
                }

                if (r) {
                    upper[i] = sqrt(pointCenterDist2(i, closest));

                    r = false;
                    // MTI clause III
                    if ((upper[i] <= centerCenterDistDiv2[closest * k + j])) {
                        continue;
                    }
                }

                auto dist = sqrt(pointCenterDist2(i, j));

                if (dist < upper[i]) {
                    closest = j;
                    upper[i] = dist;
                }
            }
            if (assignment[i] != closest) {
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

void MTIKmeans::update_bounds(int startNdx, int endNdx) {
    for (int i = startNdx; i < endNdx; ++i) {
        upper[i] += centerMovement[assignment[i]];
    }
}

void MTIKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    numLowerBounds = 0;
    TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);
    centerCenterDistDiv2 = new double[k * k];
    std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);
}

void MTIKmeans::free() {
    TriangleInequalityBaseKmeans::free();
    delete [] centerCenterDistDiv2;
    centerCenterDistDiv2 = NULL;
}

