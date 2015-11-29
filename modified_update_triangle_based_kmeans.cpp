/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
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
    delete [] neighbours;
    oldCenters = NULL;
    lowerBoundUpdate = NULL;
    maxUpperBound = NULL;
    oldCentroidsNorm2 = NULL;
    centroidsNorm2 = NULL;
    oldNewCentroidInnerProduct = NULL;
    centerCenterDistDiv2 = NULL;
    neighbours = NULL;
}

/* This method moves the newCenters to their new locations, based on the
 * sufficient statistics in sumNewCenters. It also computes the centerMovement
 * and the center that moved the furthest.
 *
 * Here the implementation adds the lower bound update. This includes copying
 * the old centroid locations and calculating the update. After this method
 * the center-center distances are known, the s array is filled and the tighter
 * lower bound update is calculated.
 *
 * Parameters: none
 *
 * Return value: none
 */
void ModifiedUpdateTriangleBasedKmeans::move_centers(int threadId)
{
    // copy the location of centers into oldCenters so that we know it
    if(threadId == 0)
    {
        memcpy(oldCenters->data, centers->data, sizeof(double) * (d * k));

        // move the centers as we do in the original algorithms
        int furthestMovingCenter = TriangleInequalityBaseKmeans::move_centers();
        converged = (0.0 == centerMovement[furthestMovingCenter]);
    }
    synchronizeAllThreads();

    // if not converged ...
    if(!converged)
    {
        // ... calculate the lower bound update
        update_s(threadId); // first update center-center distances
        calculate_max_upper_bound(threadId); // get m(c_i)
        update_cached_inner_products(threadId); // update precalculated norms
        synchronizeAllThreads();
        calculate_lower_bound_update(threadId); // and get the update
    }

//    if(threadId == 0)
//        std::cerr << lowerBoundUpdate[0] << std::endl;
}

void ModifiedUpdateTriangleBasedKmeans::update_cached_inner_products(int threadId)
{
    // copy the oldCentroids norms, we need to store this value
#ifdef USE_THREADS
    if(threadId == 0)
#endif
        memcpy(oldCentroidsNorm2, centroidsNorm2, sizeof(double) * k);
    synchronizeAllThreads();

    for (int c = 0; c < k; ++c) // for each centroid
#ifdef USE_THREADS
        if(c % numThreads == threadId)
#endif
        {
            int offset = c*d;
            // calculate norm of each centroid
            centroidsNorm2[c] = inner_product(centers->data + offset, centers->data + offset);
            // calculate the inner product of new and old location
            oldNewCentroidInnerProduct[c] = inner_product(centers->data + offset, oldCenters->data + offset);
        }

    // sort centers by movemenet, use function center_movement_comparator_function as comparator
#ifdef USE_THREADS
    if(threadId == 0)
#endif
        std::sort(centersByMovement.begin(), centersByMovement.end(), std::bind(&ModifiedUpdateTriangleBasedKmeans::center_movement_comparator_function, this, std::placeholders::_1, std::placeholders::_2));
}

/* This method does the aggregation of the updates so that we can
 * use it in algorithms with single bound. It is overridden in modified
 * versions of Elkan's algorithm.
 */
void ModifiedUpdateTriangleBasedKmeans::calculate_lower_bound_update(int threadId)
{
    // big C is the point for that we calculate the update
    for (int C = 0; C < k; ++C)
#ifdef USE_THREADS
        if(C % numThreads == threadId)
#endif
        {
            calculate_neighbors(C);

            double maxUpdate = 0;

            // iterate in decreasing order of centroid movement
            // ... it is already stored this way in neighbors array
            for (int* ptr = neighbours[C]; (*ptr) != -1; ++ptr)
            {
                // and small c is the other point that moved
                const int c = (*ptr);

                // if all remaining centroids moved less than the current update, we do not
                // need to consider them - the case of Hamerly's & heap
                if(centerMovement[c] <= maxUpdate)
                    break;

                // calculate update and overwrite if it is bigger than the current value
                double update = calculate_update(C, c);
                if(update > maxUpdate)
                    maxUpdate = update;
            }

            // store the tighter update for this cluster
            lowerBoundUpdate[C] = maxUpdate;
        }
}

// notion: C is the point c_i in the thesis c is the point c_j in the thesis

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

    // calculate the distance ||P(c_i)-c_i|| using the extended form, now only square
    double distanceOfCFromLine = cNorm2 * (1 - factor) * (1 - factor)
        + ccPrimeInnerProduct * 2 * factor * (1 - factor)
        - cCInnerProduct * 2 * (1 - factor)
        - 2 * factor * cPrimeCInnerProduct
        + CNorm2
        + factor * factor * cPrimeNorm2;
    // rounding errors make this sometimes negative if the distance is low
    // then this sqrt causes NaN - do abs value therefore
    // ... from the definition this should never happen
    if(distanceOfCFromLine < 0)
        distanceOfCFromLine = -distanceOfCFromLine;
    // calculate the distance
    distanceOfCFromLine = sqrt(distanceOfCFromLine);

    // project the y coordinate
    double y = 1 - factor * 2;
    // project the radius of the sphere around the cluster
    double r = 2 * maxUpperBoundC / cMovement;

    // here we will store the update divided by cMovement
    double update;
    // the case when the sphere with radius r around C goes through c-c' line
    if(distanceOfCFromLine < maxUpperBoundC)
    {
        // take the bottommost point where the sphere can be = bound by hyperplane
        // perpendicular to the line through c and c'
        update = r - y;
        if(update >= 1.0) // this is too bad, triangle inequality gives us better result
            return cMovement;
            // put there zero, because sphere can be curved less than the hyperbola and therefore
            // condition that while circle is above the hyperbola may be invalid
            // bound therefore by a hyperplane that goes throught the origin and is perpendicular to c-c'
        else if(update < 0)
            return 0.0; // we do not need to scale zero by multiplying, return here
        return update * cMovement;
    }

    // this is the cae when the circle contains no point with a negative coordinate
    if(y > r)
    {
        if(!consider_negative)
            return 0.0;
        // Note that if the bottommost point of the sphere has y-coordinate
        // from interval [0,1], we have to use update 0
        if(y - r <= 1.0)
            return 0.0;

        // handle the negative update ... it is the same as decreasing y by 1,
        // i.e. moving sphere by half center movement down
        y -= 1.0;
    }

    // the case that fulfils conditions of Lemma 3.1 (or the case when y = y-1)
    // encode the formula
    const double x = 2 * distanceOfCFromLine / cMovement;
    const double xSqPlusYSq = x * x + y*y;
    const double aNorm = sqrt(xSqPlusYSq - r * r);
    update = (x * r - y * aNorm) / xSqPlusYSq;

    // multiply back by the movement of c
    return update * cMovement;
}

void ModifiedUpdateTriangleBasedKmeans::calculate_max_upper_bound(int threadId)
{
    if(threadId == 0)
    {
        // calculate over the array of upper bound and find maximum for each cluster
        std::fill(maxUpperBound, maxUpperBound + k, 0.0);
        // TODO parralelize this
        for (int i = 0; i < n; ++i)
            if(maxUpperBound[assignment[i]] < upper[i])
                maxUpperBound[assignment[i]] = upper[i];
    }
}

bool ModifiedUpdateTriangleBasedKmeans::center_movement_comparator_function(int c1, int c2)
{
    return(centerMovement[c1] > centerMovement[c2]); // values must be decreaing
}

// copied from elkan_kmeans.cpp
// finds the center-center distances

void ModifiedUpdateTriangleBasedKmeans::update_s(int threadId)
{
    // find the inter-center distances
    for (int c1 = 0; c1 < k; ++c1)
#ifdef USE_THREADS
        if(c1 % numThreads == threadId)
#endif
        {
            s[c1] = std::numeric_limits<double>::max();

            for (int c2 = 0; c2 < k; ++c2)
                if(c1 != c2)
                {
                    centerCenterDistDiv2[c1 * k + c2] = sqrt(centerCenterDist2(c1, c2)) / 2.0;

                    if(centerCenterDistDiv2[c1 * k + c2] < s[c1])
                        s[c1] = centerCenterDistDiv2[c1 * k + c2];
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

    // set length k if heap (numLB = 0), otherwise k*numLowerBounds
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
        // let centers by movement contain initially 0-1-2-3-...-k, it will be sorted each iteration
        centersByMovement.push_back(c);
    }

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

/* Here we will calculate the neighbors of the cluster centered at C. */
void ModifiedUpdateTriangleBasedKmeans::calculate_neighbors(const int C)
{
    // This is half of the bound on distance between center C and some
    // other c. If || C-c || is greater than twice this, then c is not
    // neighbor of C.
    double boundOnOtherDistance = maxUpperBound[C] + s[C] + centerMovement[C];

    // find out which clusters are neighbours of cluster C
    int neighboursPos = 0;
    for (int i = 0; i < k; ++i)
    {
        // let them be sorted by movement, we need to go through in this order in the second loop
        // so as to eliminate the updates calculations for Hamerly & heap
        int c = centersByMovement[i];
        // exclude also C from negihbors and check the condition
        if(c != C && boundOnOtherDistance >= centerCenterDistDiv2[C * k + c])
            neighbours[C][neighboursPos++] = c;
    }
    // place the stop mark
    neighbours[C][neighboursPos] = -1;
}
