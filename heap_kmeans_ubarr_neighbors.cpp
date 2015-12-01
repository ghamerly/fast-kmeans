/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "heap_kmeans_ubarr_neighbors.h"
#include "general_functions.h"
#include <cmath>
#include <algorithm>

int HeapKmeansUBarrNeighbors::runThread(int threadId, int maxIterations)
{
	int iterations = 0;

	std::greater<std::pair<double, int>> heapComp;

	update_s(threadId);
    synchronizeAllThreads();

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

				double u = upper[i] + ubHeapBounds[closest];
				const double originalLower = bound + u;

				if(u <= s[closest]){
					const double newLower = heapBounds[closest] + 2 * (s[closest] - u);
					heap.push_back(std::make_pair(newLower, i));
					std::push_heap(heap.begin(), heap.end(), heapComp);
					continue;
				}

				double u2 = pointCenterDist2(i, closest);
				u = sqrt(u2);

				if(u <= std::max(s[closest], originalLower) && iterations != 1)
				{
					upper[i] = u - ubHeapBounds[closest];
					const double newLowerUpper = heapBounds[closest] + std::max(originalLower, 2 * s[closest] - u) - u;
					heap.push_back(std::make_pair(newLowerUpper, i));
					std::push_heap(heap.begin(), heap.end(), heapComp);
					continue;
				}
                else if(iterations == 1 && u < s[closest]) {
					upper[i] = u - ubHeapBounds[closest];
					Heap &newHeap = heaps[threadId][closest];
					const double newLower = heapBounds[closest] + 2 * (s[closest] - u);
					newHeap.push_back(std::make_pair(newLower, i));
					std::push_heap(newHeap.begin(), newHeap.end(), heapComp);

					Heap &ubHeap = maxUBHeap[threadId][closest];
					ubHeap.push_back(std::make_pair(u - ubHeapBounds[closest], i));
					std::push_heap(ubHeap.begin(), ubHeap.end());
					continue;
				}

                double l2 = std::numeric_limits<double>::max();

                // this is the difference from the parent class, we iterate only over selected centroids here
				for(int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr)
				{
                    // now j has the same meaning as in the parent class
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

				if((bound == 0.0) && (nextClosest < closest))
				{
					closest = nextClosest;
				}

				if(closest != assignment[i] || iterations == 1)
				{
					Heap &ubHeap = maxUBHeap[threadId][closest];
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
		move_centers(threadId);

        synchronizeAllThreads();
        if(threadId == 0 && !converged)
			update_bounds();

		synchronizeAllThreads();
	}

	return iterations;
}
