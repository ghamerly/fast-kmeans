#ifndef MODIFIED_UPDATE_TRIANGLE_BASED_KMEANS_H
#define MODIFIED_UPDATE_TRIANGLE_BASED_KMEANS_H

/* Authors: Greg Hamerly and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * ModifiedUpdateTriangleBasedKmeans is a base class for kmeans algorithms
 * that want to benefit from the tigher bound update. Goal of this class
 * is to provide infrastructure that allows efficient calculations of this
 * tigher update.
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

	/* This function claculated inner product of two vectors of dimension d.
	 * The number of inner products is counted in the superclass.
	 */
	virtual inline double inner_product(double *a, double const *b) {
#ifdef COUNT_DISTANCES
		++numInnerProducts;
#endif
		return std::inner_product(a, a + d, b, 0.0);
	};


protected:
	virtual void update_s(int threadId);

	/* Override how the centers are moved. Before the movement, we need to
	 * backup the old location of centroids and after movement we need to
	 * calculate the tigher upper bound update.
	 */
	virtual int move_centers();

	/* Calculates the lower bound update assuming that all the cached values
	 * are properly calculated. It assumes a single lower bound and therefore
	 * takes max over centroids that fulfil the neighbor condition.
     */
	virtual void calculate_lower_bound_update();

	/* Updates the cached inner products.
     */
	void update_cached_inner_products();

	/* Calculates the maximum upper bound by simple iteration over upper
	 * bound array.
	 */
	virtual void calculate_max_upper_bound();

	/* Comarator of two points by their movement. The points that have moved
	 * more are first.
	 *
	 * Parameters: the centroids to compare.
	 */
	bool center_movement_comparator_function(int c1, int c2);

	/**
	 * Calculates the tigher update.
	 *
	 * Parameters:
	 *  C the center whose update should be calculated
     *  c the center that moved
     *  consider_negative Should the calculation consider the negative value.
	 *       This option should be true for Elkan kmeans. Negative update costs
	 *       a bit more.
     * Returns: tigher update of the lower bound of points assigned to C if we
	 *       consider only centroid c
     */
	double calculate_update(const unsigned int C, const unsigned int c, bool consider_negative = false);

	/* Backup the old location of the centers. */
	Dataset *oldCenters;

	/* Array for the lower bound update. Size is set automatically using the numLowerBounds * k.
	 * When numLowerBounds is zero (heap algorithm), then this has size of k.
	 */
	double *lowerBoundUpdate;

	/* Array for storing maximum upper bound per cluster. */
	double *maxUpperBound;

	/* Norm of the centroids in the last iteration. */
	double *oldCentroidsNorm2;

	/* Norm of the current centroids. */
	double *centroidsNorm2;

	/* Inner product of each centroid's new location with old location. */
	double *oldNewCentroidInnerProduct;

	/* Keep track of the distance (divided by 2) between each pair of
	 * points. */
	double *centerCenterDistDiv2;

	/* Here we will store centroids sorted by how much they moved. */
	std::vector<int> centersByMovement;
};

#endif
