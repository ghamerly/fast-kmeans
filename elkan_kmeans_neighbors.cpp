/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "elkan_kmeans_neighbors.h"
#include "general_functions.h"
#include <cmath>

int ElkanKmeansNeighbors::runThread(int threadId, int maxIterations)
{
	int iterations = 0;

	int startNdx = start(threadId);
	int endNdx = end(threadId);

    // this is located elsewhere than in elkan_kmeans.cpp
	update_s(threadId);

	while((iterations < maxIterations) && !converged)
	{
		++iterations;

        // now we need to filter neighbors so that we remain only with those
        // that fulfil the stronger condition that is used for elkan_kmeans
        if(iterations != 1)
            filter_neighbors(threadId);

		synchronizeAllThreads();

		for (int i = startNdx; i < endNdx; ++i)
		{
			unsigned short closest = assignment[i];
			bool r = true;

			if(upper[i] <= s[closest])
			{
				continue;
			}

            // iterate only over centroids that can be closest to some x in the dataset
            for(int* ptr = neighbours[closest]; (*ptr) != -1; ++ptr)
			{
				const int j = (*ptr);
				if(j == closest)
				{
					continue;
				}
				if(upper[i] <= lower[i * k + j])
				{
					continue;
				}
				if(upper[i] <= centerCenterDistDiv2[closest * k + j])
				{
					continue;
				}

				// ELKAN 3(a)
				if(r)
				{
					upper[i] = sqrt(pointCenterDist2(i, closest));
					lower[i * k + closest] = upper[i];
					r = false;
					if((upper[i] <= lower[i * k + j]) || (upper[i] <= centerCenterDistDiv2[closest * k + j]))
					{
						continue;
					}
				}

				// ELKAN 3(b)
				lower[i * k + j] = sqrt(pointCenterDist2(i, j));
				if(lower[i * k + j] < upper[i])
				{
					closest = j;
					upper[i] = lower[i * k + j];
				}
			}
			if(assignment[i] != closest)
			{
				changeAssignment(i, closest, threadId);
			}
		}

		verifyAssignment(iterations, startNdx, endNdx);

		// ELKAN 4, 5, AND 6
		synchronizeAllThreads();
		if(threadId == 0)
		{
			int furthestMovingCenter = move_centers();
			converged = (0.0 == centerMovement[furthestMovingCenter]);
		}

		synchronizeAllThreads();
		if(!converged)
		{
			update_bounds(startNdx, endNdx);
		}
		synchronizeAllThreads();
	}

	return iterations;
}

void ElkanKmeansNeighbors::filter_neighbors(int threadId) {
    for (int c = 0; c < k; ++c) {
        if(c % numThreads == threadId) {
            double boundOnOtherDistance = maxUpperBound[c] + centerMovement[c];
            int neighboursPos = 0;
			for (int C = 0; C < k; ++C) {
                // select only centroids that fulfil the stronger condition
				if(c != C && boundOnOtherDistance > centerCenterDistDiv2[c * k + C])
					neighbours[c][neighboursPos++] = C;
			}
			neighbours[c][neighboursPos] = -1;
        }
    }
}