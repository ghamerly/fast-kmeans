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
 * and the center that moved the furthest.
 *
 * Parameters: none
 *
 * Return value: index of the furthest-moving centers
 */
int ModifiedUpdateTriangleBasedKmeans::move_centers()
{
	// memcopy is destination, source, length
	memcpy(oldCenters->data, centers->data, sizeof(double) * (d * k));

	int furthestMovingCenter = TriangleInequalityBaseKmeans::move_centers();

	if(centerMovement[furthestMovingCenter] != 0.0)
	{
		update_s(0);
		calculate_max_upper_bound();
		update_cached_inner_products();
		calculate_lower_bound_update();
	}

	return furthestMovingCenter;
}

void ModifiedUpdateTriangleBasedKmeans::update_cached_inner_products()
{
	memcpy(oldCentroidsNorm2, centroidsNorm2, sizeof(double) * k);

	for (int c = 0; c < k; ++c)
	{
		int offset = c*d;
		centroidsNorm2[c] = inner_product(centers->data + offset, centers->data + offset);
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
		double boundOnOtherDistance = maxUpperBound[C] + s[C] + centerMovement[C];

		// and small c is the other point that moved
		for (int i = 0; i < k; ++i)
		{
			int c = centersByMovement[i];

			if(centerMovement[c] <= maxUpdate)
				break;

			if(c != C && boundOnOtherDistance >= centerCenterDistDiv2[C*k + c])
			{
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
	const double cCInnerProduct = inner_product(oldCenters->data + c * d, oldCenters->data + C * d);
	const double cPrimeCInnerProduct = inner_product(centers->data + c * d, oldCenters->data + C * d);
	const double ccPrimeInnerProduct = oldNewCentroidInnerProduct[c];
	const double cNorm2 = oldCentroidsNorm2[c];
	const double cPrimeNorm2 = centroidsNorm2[c];
	const double CNorm2 = oldCentroidsNorm2[C];

	double maxUpperBoundC = maxUpperBound[C];
	double cMovement = centerMovement[c];

	double factor = (cNorm2 - cCInnerProduct + cPrimeCInnerProduct - ccPrimeInnerProduct) / cMovement / cMovement;

	double distanceOfCFromLine = cNorm2 * (1 - factor) * (1 - factor)
		+ ccPrimeInnerProduct * 2 * factor * (1 - factor)
		- cCInnerProduct * 2 * (1 - factor)
		- 2 * factor * cPrimeCInnerProduct
		+ CNorm2
		+ factor * factor * cPrimeNorm2;
	// rounding errors make this sometimes negative if the distance is low
	// then this sqrt causes NaN - do abs value therefore
	if(distanceOfCFromLine < 0)
		distanceOfCFromLine = -distanceOfCFromLine;
	distanceOfCFromLine = sqrt(distanceOfCFromLine);

	// do not care about sign, it is the same if it is + or -
	double y = 1 - factor * 2;
	double r = 2 * maxUpperBoundC / cMovement;

	double update;
	if(distanceOfCFromLine < maxUpperBoundC)
	{
		update = r - y;
		if(update > 1)
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

/* This function initializes the upper/lower bounds, assignment, centerCounts,
 * and sumNewCenters. It sets the bounds to invalid values which will force the
 * first iteration of k-means to set them correctly.  NB: subclasses should set
 * numLowerBounds appropriately before entering this function.
 *
 * Parameters: none
 *
 * Return value: none
 */
void ModifiedUpdateTriangleBasedKmeans::initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads)
{
	TriangleInequalityBaseKmeans::initialize(aX, aK, initialAssignment, aNumThreads);

	// initialize them before calling super, because moveCenters() is overriden
	oldCenters = new Dataset(k, d);

	lowerBoundUpdate = new double[k * (numLowerBounds == 0 ? 1 : numLowerBounds)];
	maxUpperBound = new double[k];

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
		centersByMovement.push_back(c);
	}
}
