/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "elkan_kmeans_modified.h"
#include "general_functions.h"
#include <cmath>

// TODO this version of the code runs approx 20% slower than the original one, I used before
// code should be the same, the #distances etc. is the same, but the runtime is worse
// * Can extracting superclass break locality that much?

void ElkanKmeansModified::calculate_lower_bound_update()
{
	// big C is the point for that we calculate the update
	for (int C = 0; C < k; ++C)
	{
		// and small c is the other point that moved
		for (int c = 0; c < k; ++c)
		{
			if(c != C)
			{
                // calculate the update and store it on the place
				double update = 0.0;
				if(centerMovement[c] != 0.0)
					update = calculate_update(C, c, true);
				lowerBoundUpdate[C * k + c] = update;
			}
		}
	}
}

void ElkanKmeansModified::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads)
{
	numLowerBounds = aK;
	ModifiedUpdateTriangleBasedKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

	std::fill(lowerBoundUpdate, lowerBoundUpdate + k * k, 0.0);
}

int ElkanKmeansModified::runThread(int threadId, int maxIterations)
{
	int iterations = 0;

	int startNdx = start(threadId);
	int endNdx = end(threadId);

    // this is located elsewhere than in elkan_kmeans.cpp
	update_s(threadId);

	while((iterations < maxIterations) && !converged)
	{
		++iterations;

		synchronizeAllThreads();

		for (int i = startNdx; i < endNdx; ++i)
		{
			unsigned short closest = assignment[i];
			bool r = true;

			if(upper[i] <= s[closest])
			{
				continue;
			}

			for (int j = 0; j < k; ++j)
			{
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

void ElkanKmeansModified::update_bounds(int startNdx, int endNdx)
{
#ifdef COUNT_DISTANCES
	for (int i = 0; i < k; ++i)
		for (int j = 0; j < numLowerBounds; ++j)
		{
			boundsUpdates += ((double) clusterSize[0][i]) * (lowerBoundUpdate[i * numLowerBounds + j]);
		}
#endif
	for (int i = startNdx; i < endNdx; ++i)
	{
		upper[i] += centerMovement[assignment[i]];
		for (int j = 0; j < numLowerBounds; ++j)
		{
            // each lower bound is updated by specified value
			lower[i * numLowerBounds + j] -= lowerBoundUpdate[assignment[i] * numLowerBounds + j];
		}
	}
}