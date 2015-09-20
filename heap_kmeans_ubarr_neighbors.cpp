/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "heap_kmeans_ubarr_neighbors.h"
#include "general_functions.h"
#include <cmath>
#include <algorithm>

int HeapKmeansUBarrNeighbors::runThread(int threadId, int maxIterations)
{
	int iterations = 0;

	std::greater<std::pair<double, int>> heapComp;

	update_s(0);

	while((iterations < maxIterations) && !converged)
	{
		++iterations;

		for (int h = 0; h < k; ++h)
		{
			Heap &heap = heaps[threadId][h];

			while(heap.size() > 0)
			{
				if(heapBounds[h] <= heap[0].first)
				{
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
				if(u <= s[closest]){ // note that this cannot happen in the 1st iteration (u = inf)
					const double newLower = heapBounds[closest] + 2 * (s[closest] - u);
					heap.push_back(std::make_pair(newLower, i));
					std::push_heap(heap.begin(), heap.end(), heapComp);
					continue;
				}

				double u2 = pointCenterDist2(i, closest);
				u = sqrt(u2);

                // same condition as in the case of Hamerly's algorithm
				if(u <= std::max(s[closest], originalLower) && iterations != 1)
				{
					upper[i] = u - ubHeapBounds[closest];
					const double newLowerUpper = heapBounds[closest] + std::max(originalLower, 2 * s[closest] - u) - u;
					heap.push_back(std::make_pair(newLowerUpper, i));
					std::push_heap(heap.begin(), heap.end(), heapComp);
					continue;
				}
                // in the case of first iteration, store the upper bound in the heap
                // using s[closest] as estimate, if possible
                else if(iterations == 1 && u < s[closest]) {
					upper[i] = u - ubHeapBounds[closest];
					Heap &newHeap = heaps[threadId][closest];
					const double newLower = heapBounds[closest] + 2 * (s[closest] - u);
					newHeap.push_back(std::make_pair(newLower, i));
					std::push_heap(newHeap.begin(), newHeap.end(), heapComp);

                    // we have to store into the maxUBHeap as we need to ensure that
                    // *all* points are in this heap with at least some upper bound
					Heap &ubHeap = maxUBHeap[closest];
					ubHeap.push_back(std::make_pair(u - ubHeapBounds[closest], i));
					std::push_heap(ubHeap.begin(), ubHeap.end());
					continue;
				}

                double l2 = std::numeric_limits<double>::max();

				for(int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr)
				{
                    int j = (*ptr);
					if(j == closest) continue;

					double dist2 = pointCenterDist2(i, j);
					if(dist2 < u2)
					{
						l2 = u2;
						u2 = dist2;
						nextClosest = closest;
						closest = j;
					}
					else if(dist2 < l2)
					{
						l2 = dist2;
						nextClosest = j;
					}
				}

				u = sqrt(u2);
				bound = sqrt(l2) - u;

				// Break ties consistently with Lloyd (also prevents infinite cycle)
				if((bound == 0.0) && (nextClosest < closest))
				{
					closest = nextClosest;
				}

				if(closest != assignment[i] || iterations == 1) // iterations == 1 needed for points that are assigned correctly
				{
                    // we need to make sure that all points are in ubHeap after 1st iteration
                    // or after the assignment change
					Heap &ubHeap = maxUBHeap[closest];
					ubHeap.push_back(std::make_pair(u - ubHeapBounds[closest], i));
					std::push_heap(ubHeap.begin(), ubHeap.end());
				}

				if(closest != assignment[i])
				{
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
		if(threadId == 0)
		{
			int furthestMoving = move_centers();
			converged = (0.0 == centerMovement[furthestMoving]);
			update_bounds();
		}

		synchronizeAllThreads();
	}

	return iterations;
}
