/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "modified_update_triangle_based_kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <functional>

void ModifiedUpdateTriangleBasedKmeans::free()
{
	TriangleInequalityBaseKmeans::free();
	centersByMovement.clear();
	delete oldCenters;
	delete [] lowerBoundUpdate;
	delete [] maxUpperBound;
	delete [] oldCentroidsNorm2;
	delete [] centroidsNorm2;
	delete [] oldNewCentroidInnerProduct;
	delete [] centerCenterDistDiv2;
	oldCenters = NULL;
	lowerBoundUpdate = NULL;
	maxUpperBound = NULL;
	oldCentroidsNorm2 = NULL;
	centroidsNorm2 = NULL;
	oldNewCentroidInnerProduct = NULL;
	centerCenterDistDiv2 = NULL;
}

/* This method moves the newCenters to their new locations, based on the
 * sufficient statistics in sumNewCenters. It also computes the centerMovement
 * and the center that moved the furthest. Here the implementation adds the
 * loewer bound update.
 *
 * Parameters: none
 *
 * Return value: index of the furthest-moving centers
 */
int ModifiedUpdateTriangleBasedKmeans::move_centers()
{
	// copy the location of centers into oldCenters so that we know it
	memcpy(oldCenters->data, centers->data, sizeof(double) * (d * k));

    // move the centers
	int furthestMovingCenter = TriangleInequalityBaseKmeans::move_centers();

    // if not converged ...
	if(centerMovement[furthestMovingCenter] != 0.0)
	{
        // ... calculate the lower bound update
		update_s(0);
		calculate_max_upper_bound();
		update_cached_inner_products();
		calculate_lower_bound_update();
	}

	return furthestMovingCenter;
}

void ModifiedUpdateTriangleBasedKmeans::update_cached_inner_products()
{
    // copy the oldCentroids norms, we need to store this value
	memcpy(oldCentroidsNorm2, centroidsNorm2, sizeof(double) * k);

	for (int c = 0; c < k; ++c)
	{
		int offset = c*d;
        // calculate norm of each centroid
		centroidsNorm2[c] = inner_product(centers->data + offset, centers->data + offset);
        // calculate the inner product of new and old location
		oldNewCentroidInnerProduct[c] = inner_product(centers->data + offset, oldCenters->data + offset);
	}

	// sort centers by movemenet, use function center_movement_comparator_function as comparator
	std::sort(centersByMovement.begin(), centersByMovement.end(), std::bind(&ModifiedUpdateTriangleBasedKmeans::center_movement_comparator_function, this, std::placeholders::_1, std::placeholders::_2));
}

void ModifiedUpdateTriangleBasedKmeans::calculate_lower_bound_update()
{
	// big C is the point for that we calculate the update
	for (int C = 0; C < k; ++C)
	{
		double maxUpdate = 0;
        // see triangle_based_kmeans_neighbors for meaning of this (condition for neighbor)
		double boundOnOtherDistance = maxUpperBound[C] + s[C] + centerMovement[C];

		// and small c is the other point that moved
		for (int i = 0; i < k; ++i)
		{
            // iterate in decreasing order of centroid movement
			int c = centersByMovement[i];

            // if all remaining centroids moved less than the current update, we do not
            // need to consider them - the case of Hamerly & heap
			if(centerMovement[c] <= maxUpdate)
				break;

            // if centroid c is a neighbor of C, we need to consider it for update
			if(c != C && boundOnOtherDistance >= centerCenterDistDiv2[C*k + c])
			{
                // calculate update and overwrite if it is bigger than the current value
				double update = calculate_update(C, c);
				if(update > maxUpdate)
					maxUpdate = update;
			}
		}

		lowerBoundUpdate[C] = maxUpdate;
	}
}

double ModifiedUpdateTriangleBasedKmeans::calculate_update(const unsigned int C, const unsigned int c, bool consider_negative)
{
    // those values will be needed
	const double cCInnerProduct = inner_product(oldCenters->data + c * d, oldCenters->data + C * d);
	const double cPrimeCInnerProduct = inner_product(centers->data + c * d, oldCenters->data + C * d);
	const double ccPrimeInnerProduct = oldNewCentroidInnerProduct[c];
	const double cNorm2 = oldCentroidsNorm2[c];
	const double cPrimeNorm2 = centroidsNorm2[c];
	const double CNorm2 = oldCentroidsNorm2[C];

	double maxUpperBoundC = maxUpperBound[C];
	double cMovement = centerMovement[c];

    // t, that specifies the projection of C onto c cPrime line (P(c_i) = c_j + t * (c_j' - c_j))
	double factor = (cNorm2 - cCInnerProduct + cPrimeCInnerProduct - ccPrimeInnerProduct) / cMovement / cMovement;

    // calculate the distance using the extended form, now only square
	double distanceOfCFromLine = cNorm2 * (1 - factor) * (1 - factor)
		+ ccPrimeInnerProduct * 2 * factor * (1 - factor)
		- cCInnerProduct * 2 * (1 - factor)
		- 2 * factor * cPrimeCInnerProduct
		+ CNorm2
		+ factor * factor * cPrimeNorm2;
	// rounding errors make this sometimes negative if the distance is low
	// then this sqrt causes NaN - do abs value therefore
    // ... from the definition this shoul never happen
	if(distanceOfCFromLine < 0)
		distanceOfCFromLine = -distanceOfCFromLine;
    // calculate the distance
	distanceOfCFromLine = sqrt(distanceOfCFromLine);

	// do not care about sign, it is the same if it is + or -
	double y = 1 - factor * 2;
	double r = 2 * maxUpperBoundC / cMovement;

    // update divided by cMovement
	double update;
    // the case when the sphere with radius r around C goes through c-c' line
	if(distanceOfCFromLine < maxUpperBoundC)
	{
        // take the bottommost point where the sphere can be = bound by hyperplane
		update = r - y;
		if(update > 1) // this is not necessary, triangle inequality is enough
			update = 1;
		// put there zero, because sphere can be curved less than hyperbola and therefore
		// negative condition may be invalid ... be careful about this
		else if(update < 0) // the area betwenn c and center is prohibited
			update = 0;
	}
	else
	{
		double x = 2 * distanceOfCFromLine / cMovement;
		double xSqPlusYSq = x * x + y*y;
		double aNorm = sqrt(xSqPlusYSq - r * r);
		update = (x * r - y * aNorm) / xSqPlusYSq;

		// handle the negative update ... it is the same as decreasing y by 1,
		// i.e. moving sphere by half center movement down
		if(consider_negative && update < 0)
		{
			y -= 1;
			xSqPlusYSq = x * x + y*y;
			aNorm = sqrt(xSqPlusYSq - r * r);
			update = (x * r - y * aNorm) / xSqPlusYSq;
		}
	}

    // multiply back by the movement of c
	return update * cMovement;
}

void ModifiedUpdateTriangleBasedKmeans::calculate_max_upper_bound()
{
	std::fill(maxUpperBound, maxUpperBound + k, 0.0);
	for (int i = 0; i < n; ++i)
		if(maxUpperBound[assignment[i]] < upper[i])
			maxUpperBound[assignment[i]] = upper[i];
}

bool ModifiedUpdateTriangleBasedKmeans::center_movement_comparator_function(int c1, int c2)
{
	return(centerMovement[c1] > centerMovement[c2]); // values must be decreaing
}

// copied from Elkan kmeans
void ModifiedUpdateTriangleBasedKmeans::update_s(int threadId)
{
	// find the inter-center distances
	for (int c1 = 0; c1 < k; ++c1)
	{
		if(c1 % numThreads == threadId)
		{
			s[c1] = std::numeric_limits<double>::max();

			for (int c2 = 0; c2 < k; ++c2)
			{
				if(c1 != c2) {
                    // divide by 2 here since we always use the inter-center
                    // distances divided by 2
                    centerCenterDistDiv2[c1 * k + c2] = sqrt(centerCenterDist2(c1, c2)) / 2.0;

                    if(centerCenterDistDiv2[c1 * k + c2] < s[c1])
                    {
                        s[c1] = centerCenterDistDiv2[c1 * k + c2];
                    }
                }
			}
		}
	}
}

/* This function initializes the old centers, bound update, maxUpperBound,
 * cached inner product, the vector of centroids sorted by movement, distances
 * between centroids and also norms of centroids.
 *
 * Parameters: none
 *
 * Return value: none
 */
void ModifiedUpdateTriangleBasedKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads)
{
	TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

	oldCenters = new Dataset(k, d);

    // set length k if heap, otherwise k*numLowerBounds
	lowerBoundUpdate = new double[k * (numLowerBounds == 0 ? 1 : numLowerBounds)];
	maxUpperBound = new double[k];

    // the cached inner products
	oldCentroidsNorm2 = new double[k];
	centroidsNorm2 = new double[k];
	oldNewCentroidInnerProduct = new double[k];

	centerCenterDistDiv2 = new double[k * k];
	std::fill(centerCenterDistDiv2, centerCenterDistDiv2 + k * k, 0.0);

	for (int c = 0; c < k; ++c)
	{
		int offset = c*d;
		// calcualte norms of initial centers
		centroidsNorm2[c] = inner_product(centers->data + offset, centers->data + offset);
        // let centers by movement contain 0-1-2-3-...-k, it will be sorted
		centersByMovement.push_back(c);
	}
}
