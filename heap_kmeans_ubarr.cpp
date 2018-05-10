/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "heap_kmeans_ubarr.h"
#include "general_functions.h"
#include <cmath>
#include <algorithm>

void HeapKmeansUBarr::free() {
    HeapKmeansModified::free();
    for (int t = 0; t < numThreads; ++t) {
        delete [] maxUBHeap[t];
    }
    delete[] maxUBHeap;
    delete[] ubHeapBounds;
    maxUBHeap = NULL;
    ubHeapBounds = NULL;
}

/* Calculates the maximum upper bound over each cluster. This is achieved by
 * looking onto the top of the heaps that contain the upper bound.
 */
void HeapKmeansUBarr::calculate_max_upper_bound(int threadId) {
    for (int c = 0; c < k; ++c) {
        Heap &heap = maxUBHeap[threadId][c];
        while (heap.size() > 1) {
            // look onto the top of the heap
            // check that the point is still assigned to this cluter
            if (c == assignment[heap[0].second]) {
                // also the upper bound may be invalid - it may have been tigtened
                if (upper[heap[0].second] == heap[0].first)
                    break;

                // if the entry is invalid, push a new one with the correct upper bound
                heap.push_back(std::make_pair(upper[heap[0].second], heap[0].second));
                std::push_heap(heap.begin(), heap.end());
            }

            // drop the outdated entries
            std::pop_heap(heap.begin(), heap.end());
            heap.pop_back();
        }

        // and do not forget that the upper bound is relative to the center movement history
        #ifdef USE_THREADS
        // remember that in multithreaded version the size of the heap may be zero
        maxUpperBoundAgg[threadId * k + c] = heap.empty() ? -1.0 : heap[0].first + ubHeapBounds[c];
        #else
        maxUpperBound[c] = heap[0].first + ubHeapBounds[c];
        #endif
    }

    #ifdef USE_THREADS
    if (threadId == 0)
        std::fill(maxUpperBound, maxUpperBound + k, 0.0);
    synchronizeAllThreads();
    aggregate_maximum_upper_bound(threadId);
    #endif
}

void HeapKmeansUBarr::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads) {
    HeapKmeansModified::initialize(aX, aK, initialAssignment, aNumThreads);

    maxUBHeap = new Heap*[numThreads];
    ubHeapBounds = new double[k];

    for (int t = 0; t < numThreads; ++t) {
        maxUBHeap[t] = new Heap[k];
    }
    std::fill(ubHeapBounds, ubHeapBounds + k, 0.0);
}

int HeapKmeansUBarr::runThread(int threadId, int maxIterations) {
    int iterations = 0;

    std::greater<std::pair<double, int>> heapComp;

    update_s(threadId);
    synchronizeAllThreads();

    while ((iterations < maxIterations) && !converged) {
        ++iterations;

        for (int h = 0; h < k; ++h) {
            Heap &heap = heaps[threadId][h];

            while (heap.size() > 0) {
                if (heapBounds[h] <= heap[0].first) {
                    break;
                }

                int i = heap[0].second;
                unsigned short closest = assignment[i];
                unsigned short nextClosest = 0;
                double bound = heap[0].first - heapBounds[closest];

                std::pop_heap(heap.begin(), heap.end(), heapComp);
                heap.pop_back();

                // calculate the real value of the upper bound
                double u = upper[i] + ubHeapBounds[closest];
                const double originalLower = bound + u;

                // if the upper bound is less than s, closest, push the point to the heap
                // with new key - using s[closest] to estimate the lower bound
                // this is equivalent to tightening the upper bound in Hamerly's aglorithm
                if (u <= s[closest]) { // note that this cannot happen in the 1st iteration (u = inf)
                    const double newLower = heapBounds[closest] + 2 * (s[closest] - u);
                    heap.push_back(std::make_pair(newLower, i));
                    std::push_heap(heap.begin(), heap.end(), heapComp);
                    continue;
                }

                double u2 = pointCenterDist2(i, closest);
                u = sqrt(u2);

                // the same condition as in the case of Hamerly's algorithm
                if (u <= std::max(s[closest], originalLower) && iterations != 1) {
                    upper[i] = u - ubHeapBounds[closest];
                    const double newLowerUpper = heapBounds[closest] + std::max(originalLower, 2 * s[closest] - u) - u;
                    heap.push_back(std::make_pair(newLowerUpper, i));
                    std::push_heap(heap.begin(), heap.end(), heapComp);
                    continue;
                }                    // in the case of first iteration, store the upper bound in the heap
                    // using s[closest] as estimate, if possible
                else if (iterations == 1 && u < s[closest]) {
                    upper[i] = u - ubHeapBounds[closest];
                    Heap &newHeap = heaps[threadId][closest];
                    const double newLower = heapBounds[closest] + 2 * (s[closest] - u);
                    newHeap.push_back(std::make_pair(newLower, i));
                    std::push_heap(newHeap.begin(), newHeap.end(), heapComp);

                    // we have to store into the maxUBHeap as we need to ensure that
                    // *all* points are in this heap with at least some upper bound
                    Heap &ubHeap = maxUBHeap[threadId][closest];
                    ubHeap.push_back(std::make_pair(u - ubHeapBounds[closest], i));
                    std::push_heap(ubHeap.begin(), ubHeap.end());
                    continue;
                }

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

                u = sqrt(u2);
                bound = sqrt(l2) - u;

                if ((bound == 0.0) && (nextClosest < closest)) {
                    closest = nextClosest;
                }

                if (closest != assignment[i] || iterations == 1) // iterations == 1 needed for points that are assigned correctly
                {
                    // we need to make sure that all points are in ubHeap after 1st iteration
                    // or after the assignment change
                    Heap &ubHeap = maxUBHeap[threadId][closest];
                    ubHeap.push_back(std::make_pair(u - ubHeapBounds[closest], i));
                    std::push_heap(ubHeap.begin(), ubHeap.end());
                }

                if (closest != assignment[i]) {
                    changeAssignment(i, closest, threadId);
                }

                upper[i] = u - ubHeapBounds[closest];
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

void HeapKmeansUBarr::update_bounds() {
    #ifdef COUNT_DISTANCES
    for (int i = 0; i < k; ++i)
        boundsUpdates += ((double) clusterSize[0][i]) * (lowerBoundUpdate[i]);
    #endif
    for (int j = 0; j < k; ++j) {
        // this is by what grow the upper bounds
        ubHeapBounds[j] += centerMovement[j];
        heapBounds[j] += centerMovement[j];
        heapBounds[j] += lowerBoundUpdate[j];
    }
}
