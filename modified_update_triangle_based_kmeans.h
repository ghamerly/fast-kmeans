#ifndef MODIFIED_UPDATE_TRIANGLE_BASED_KMEANS_H
#define MODIFIED_UPDATE_TRIANGLE_BASED_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * OriginalSpaceKmeans is a base class for other algorithms that operate in the
 * same space as the data being clustered (as opposed to kernelized k-means
 * algorithms, which operate in kernel space).
 */

#include "triangle_inequality_base_kmeans.h"
#include <numeric>
#include <vector>

/* Cluster with the cluster centers living in the original space (with the
 * data). This is as opposed to a kernelized version of k-means, where the
 * center points might not be explicitly represented. This is also an abstract
 * class.
 */
class ModifiedUpdateTriangleBasedKmeans : public TriangleInequalityBaseKmeans {
public:

	ModifiedUpdateTriangleBasedKmeans() : oldCenters(NULL),
	lowerBoundUpdate(NULL), maxUpperBound(NULL), oldCentroidsNorm2(NULL), centroidsNorm2(NULL),
	oldNewCentroidInnerProduct(NULL), centerCenterDistDiv2(NULL) {
	}

	virtual ~ModifiedUpdateTriangleBasedKmeans() {
		free();
	}

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
	virtual void free();

	virtual inline double inner_product(double *a, double const *b) {
#ifdef COUNT_DISTANCES
		++numInnerProducts;
#endif
		return std::inner_product(a, a + d, b, 0.0);
	};


protected:
	virtual void update_s(int threadId);

	virtual int move_centers();

	virtual void calculate_lower_bound_update();

	void update_cached_inner_products();

	virtual void calculate_max_upper_bound();

	bool center_movement_comparator_function(int c1, int c2);

	double calculate_update(const unsigned int C, const unsigned int c, bool consider_negative = false);

	// the old centroids used for calculation of the
	Dataset *oldCenters;

	double *lowerBoundUpdate;

	double *maxUpperBound;

	double *oldCentroidsNorm2;

	double *centroidsNorm2;

	double *oldNewCentroidInnerProduct;

	// Keep track of the distance (divided by 2) between each pair of
	// points.
	double *centerCenterDistDiv2;

	std::vector<int> centersByMovement;
};

#endif
