/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "hamerly_kmeans_neighbors_only.h"
#include "general_functions.h"
#include <cmath>

/* This method moves the newCenters to their new locations, based on the
 * sufficient statistics in sumNewCenters. It also computes the centerMovement
 * and the center that moved the furthest. Here the implementation adds the
 * loewer bound update.
 *
 * Parameters: none
 *
 * Return value: index of the furthest-moving centers
 */
int HamerlyKmeansNeighborsOnly::move_centers()
{
    // move the centers
	int furthestMovingCenter = TriangleInequalityBaseKmeans::move_centers();

    // if not converged ...
	if(centerMovement[furthestMovingCenter] != 0.0)
	{
        // ... calculate the lower bound update
		update_s(0);
		calculate_max_upper_bound();
        for(int i = 0; i < k; ++i)
            calculate_neighbors(i);
		calculate_lower_bound_update();
	}

	return furthestMovingCenter;
}

/* This method fills the lower bound update array in the same manner as it
 * would be filled for the default Hamerly kmeans implementation. */
void HamerlyKmeansNeighborsOnly::calculate_lower_bound_update()
{
    int furthestMovingCenter = 0;
    double longest = centerMovement[furthestMovingCenter];
    double secondLongest = 0.0;
    for (int j = 0; j < k; ++j) {
        if (longest < centerMovement[j]) {
            secondLongest = longest;
            longest = centerMovement[j];
            furthestMovingCenter = j;
        } else if (secondLongest < centerMovement[j]) {
            secondLongest = centerMovement[j];
        }
    }

    for (int i = 0; i < k; ++i)
        lowerBoundUpdate[i] = (i == furthestMovingCenter) ? secondLongest : longest;
}