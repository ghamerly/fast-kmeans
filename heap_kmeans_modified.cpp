/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "modified_update_triangle_based_kmeans.h"
#include "general_functions.h"
#include "heap_kmeans_modified.h"
#include <cmath>
#include <algorithm>

void HeapKmeansModified::free() {
    ModifiedUpdateTriangleBasedKmeans::free();
    for (int t = 0; t < numThreads; ++t) {
        delete [] heaps[t];
    }
    delete [] heaps;
    delete [] heapBounds;
    heaps = NULL;
    heapBounds = NULL;
}

void HeapKmeansModified::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    ModifiedUpdateTriangleBasedKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

    heaps = new Heap*[numThreads];
    heapBounds = new double[k];

    for (int t = 0; t < numThreads; ++t) {
        heaps[t] = new Heap[k];
        int startNdx = start(t);
        int endNdx = end(t);
        heaps[t][0].resize(endNdx - startNdx, std::make_pair(-1.0, 0));
        for (int i = 0; i < endNdx - startNdx; ++i) {
            heaps[t][0][i].second = i + startNdx;
        }
    }

    std::fill(heapBounds, heapBounds + k, 0.0);
    // start with zeros here
    std::fill(maxUpperBound, maxUpperBound + k, 0.0);
}

int HeapKmeansModified::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    std::greater<std::pair<double, int>> heapComp;

    while ((iterations < maxIterations) && !converged) {
        ++iterations;

        for (int h = 0; h < k; ++h) {
            Heap &heap = heaps[threadId][h];

            while (heap.size() > 0) {
                if (heapBounds[h] <= heap[0].first)
                    break;

                int i = heap[0].second;

                std::pop_heap(heap.begin(), heap.end(), heapComp);
                heap.pop_back();

                unsigned short closest = assignment[i];
                unsigned short nextClosest = 0;

                double u2 = pointCenterDist2(i, closest);
                double l2 = std::numeric_limits<double>::max();

                for (unsigned short j = 0; j < k; ++j) {
                    if (j == closest) continue;

                    double dist2 = pointCenterDist2(i, j);
                    if (dist2 < u2) {
                        l2 = u2;
                        u2 = dist2;
                        nextClosest = closest;
                        closest = j;
                    } else if (dist2 < l2) {
                        l2 = dist2;
                        nextClosest = j;
                    }
                }

                const double u = sqrt(u2);
                const double bound = sqrt(l2) - u;

                if ((bound == 0.0) && (nextClosest < closest)) {
                    closest = nextClosest;
                }

                // save the maximum upper bound, should be active only in the first iteration and when assignment changes
                if (u > maxUpperBound[closest])
                    maxUpperBound[closest] = u;

                if (closest != assignment[i]) {
                    changeAssignment(i, closest, threadId);
                }

                Heap &newHeap = heaps[threadId][closest];
                newHeap.push_back(std::make_pair(heapBounds[closest] + bound, i));
                std::push_heap(newHeap.begin(), newHeap.end(), heapComp);
            }
        }

        verifyAssignment(iterations, start(threadId), end(threadId));

        synchronizeAllThreads();
        move_centers(threadId);

        synchronizeAllThreads();
        if (threadId == 0 && !converged)
            update_bounds();

        synchronizeAllThreads();
    }
    return iterations;
}

void HeapKmeansModified::update_bounds() {
    #ifdef COUNT_DISTANCES
    for (int i = 0; i < k; ++i)
        boundsUpdates += ((double) clusterSize[0][i]) * (lowerBoundUpdate[i]);
    #endif
    for (int j = 0; j < k; ++j) {
        // this is the worst case by that the maximum upper bound can grow
        maxUpperBound[j] += centerMovement[j];
        heapBounds[j] += centerMovement[j];
        // update the lower bound by the calculated tighter update
        heapBounds[j] += lowerBoundUpdate[j];
    }
}

// do not do anything, we will update maximum upper bound in update_bounds

void HeapKmeansModified::calculate_max_upper_bound(int threadId) {
}
