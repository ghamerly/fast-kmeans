/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "annulus_kmeans_modified.h"
#include "general_functions.h"
#include <cmath>
#include <numeric>
#include <algorithm>

void AnnulusKmeansModified::free() {
    HamerlyKmeansModified::free();

    delete [] guard;
    delete [] xNorm;
    delete [] cOrder;
    guard = NULL;
    xNorm = NULL;
    cOrder = NULL;
}

int AnnulusKmeansModified::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    int startNdx = start(threadId);
    int endNdx = end(threadId);

    // here we need to calculate s & the centroid-centroid distances before the first iteration
    // the remaining calls to this method are hidden by move_centers
	update_s(threadId);

    while ((iterations < maxIterations) && ! converged) {
        ++iterations;

        synchronizeAllThreads();

        if (threadId == 0) {
            sort_means_by_norm();
        }
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

            double l2 = pointCenterDist2(i, guard[i]);
            lower[i] = sqrt(l2);

            double beta = std::max(lower[i], upper[i]);

            std::pair<double, int>* begin = std::lower_bound(cOrder, cOrder + k, std::make_pair(xNorm[i] - beta, k));
            std::pair<double, int>* end = std::lower_bound(begin, cOrder + k, std::make_pair(xNorm[i] + beta, k));

            for (std::pair<double, int>* jp = begin; jp != end; ++jp) {
                if (jp->second == closest) continue;

                double dist2 = pointCenterDist2(i, jp->second);
                if (dist2 <= u2) {
                    if (dist2 == u2) {
                        if (jp->second < closest) closest = jp->second;
                    } else {
                        l2 = u2;
                        u2 = dist2;
                        guard[i] = closest;
                        closest = jp->second;
                    }
                } else if (dist2 < l2) {
                    l2 = dist2;
                    guard[i] = jp->second;
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

void AnnulusKmeansModified::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    HamerlyKmeansModified::initialize(aX, aK, initialAssignment, aNumThreads);

    guard = new unsigned short[n];
    xNorm = new double[n];
    cOrder = new std::pair<double, int>[k];

    for (int i = 0; i < k; ++i) {
        cOrder[i].first = 0.0;
        cOrder[i].second = i;
    }

    std::fill(guard, guard + n, 1);
    for (int i = 0; i < n; ++i) {
        xNorm[i] = sqrt(pointPointInnerProduct(i, i));
    }
}

void AnnulusKmeansModified::sort_means_by_norm() {
    for (int c1 = 0; c1 < k; ++c1) {
        cOrder[c1].first = sqrt(centerCenterInnerProduct(c1, c1));
        cOrder[c1].second = c1;
    }
    std::sort(cOrder, cOrder + k);
}
