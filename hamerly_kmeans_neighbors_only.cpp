/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 */

#include "hamerly_kmeans_neighbors_only.h"
#include "general_functions.h"
#include <cmath>

int HamerlyKmeansNeighborsOnly::move_centers()
{
    // move the centers
	int furthestMovingCenter = TriangleInequalityBaseKmeans::move_centers();

    // if not converged ...
	if(centerMovement[furthestMovingCenter] != 0.0)
	{
        // ... calculate the neighbors and the lower bound update given by the triangle inequality
		update_s(0);
		calculate_max_upper_bound();
        for(int i = 0; i < k; ++i)
            calculate_neighbors(i);
		calculate_lower_bound_update();
	}

	return furthestMovingCenter;
}

/* This method fills the lower bound update array in the same manner as it
 * would be filled for the default Hamerly's kmeans implementation. The
 * code is copied from hamerly_kmeans.cpp with small modification. */
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

    // store the update instead of updating the bound directly
    for (int i = 0; i < k; ++i)
        lowerBoundUpdate[i] = (i == furthestMovingCenter) ? secondLongest : longest;
}