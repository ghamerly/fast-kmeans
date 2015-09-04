/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "triangle_based_kmeans_neighbors.h"
#include <cmath>

void TriangleBasedKmeansNeighbors::free()
{
	ModifiedUpdateTriangleBasedKmeans::free();
	delete [] neighbours;
	neighbours = NULL;
}

void TriangleBasedKmeansNeighbors::calculate_lower_bound_update()
{
	// big C is the point for that we calculate the update
	for (int C = 0; C < k; ++C)
	{
		double maxUpdate = 0;
        // This is half of the bound on distance between center C and some
        // other c. If || C-c || is greater than twice this, then c is not
        // neighbor of C.
		double boundOnOtherDistance = maxUpperBound[C] + s[C] + centerMovement[C];

		// find out which clusters are neighbours of cluster C
		int neighboursPos = 0;
		for (int i = 0; i < k; ++i)
		{
			int c = centersByMovement[i];
			// let them sorted by movement, we need to go through in this order in the second loop
            // so as to eliminate the updates calculations for Hamerly & heap
			if(c != C && boundOnOtherDistance >= centerCenterDistDiv2[C*k + c])
				neighbours[C][neighboursPos++] = c;
		}
        // place the stop mark
		neighbours[C][neighboursPos] = -1;

		// and small c is the other point that moved
		for(int* ptr = neighbours[C]; (*ptr) != -1; ++ptr)
		{
			int c = (*ptr);

            // if all remaining centroids moved less than the current update, we do not
            // need to consider them - the case of Hamerly & heap
			if(centerMovement[c] <= maxUpdate)
				break;

            // calculate update and overwrite if it is bigger than the current value
			double update = calculate_update(C, c);
				if(update > maxUpdate)
					maxUpdate = update;
		}

		lowerBoundUpdate[C] = maxUpdate;
	}
}

/* This function initializes the neighbours set to contain full
 * relation but \Delta.
 *
 * Parameters: none
 *
 * Return value: none
 */
void TriangleBasedKmeansNeighbors::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads)
{
	ModifiedUpdateTriangleBasedKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

	// there we will store neighbouring clusters to each cluster
	// in first iteration we have to go through all neighbours as all the upper bounds are invalid
	neighbours = new int*[k];
	for (int i = 0; i < k; ++i)
	{
		neighbours[i] = new int[k];
		int pos = 0;
		for (int j = 0; j < k; ++j)
			if(i != j)
				neighbours[i][pos++] = j;
        // place the termination sign, iteration will stop there
		neighbours[i][k - 1] = -1;
	}
}
