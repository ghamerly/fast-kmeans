#ifndef MODIFIED_UPDATE_TRIANGLE_BASED_KMEANS_H
#define MODIFIED_UPDATE_TRIANGLE_BASED_KMEANS_H

/* Authors: Greg Hamerly and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * ModifiedUpdateTriangleBasedKmeans is a base class for kmeans algorithms
 * that want to benefit from the tigher bound update, iteration over neighbors
 * and other improvements proposed in (Ryšavý, Hamerly 2015). Goal of this class
 * is to provide infrastructure that allows efficient calculations of this
 * tigher update and neighbors set.
 *
 * The idea about the tighter lower bound is the following:
 *  - If we update the lower bound using the triangle inequality, we are too
 *    pessimistic as the triangle inequality holds everywhere in the space.
 *  - Therefore we bound the cluster into a sphere.
 *  - And on the sphere we calculate the maximal update that will be needed
 *    for the lower bound considering each centroid move.
 * If we consider the neighbors set, the idea is the following:
 *  - Again we bound the cluster in a sphere.
 *  - We evaluate for each centroid a condition, which must be fulfilled if
 *    the point is the closest or the second closest to some point in the
 *    cluster.
 *  - We can ignore the poitns that violate this condition from the innermost
 *    loop in Hamerly's algorithm and also in the tighter update calculation.
 */

#include "triangle_inequality_base_kmeans.h"
#include <numeric>
#include <vector>

class ModifiedUpdateTriangleBasedKmeans : public TriangleInequalityBaseKmeans {
public:

	ModifiedUpdateTriangleBasedKmeans() : oldCenters(NULL),
	lowerBoundUpdate(NULL), maxUpperBound(NULL), maxUpperBoundAgg(NULL),
	oldCentroidsNorm2(NULL), centroidsNorm2(NULL), oldNewCentroidInnerProduct(NULL),
	centerCenterDistDiv2(NULL), neighbours(NULL) {
	}

	virtual ~ModifiedUpdateTriangleBasedKmeans() {
		free();
	}

	virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
	virtual void free();

	/* This function claculated inner product of two vectors of dimension d.
	 * The number of inner products is counted in the superclass.
	 */
	inline double inner_product(const double *a, double const *b) const {
#ifdef COUNT_DISTANCES
		++numInnerProducts;
#endif
		return std::inner_product(a, a + d, b, 0.0);
	};


protected:
	/* This method is overloaded because we use it also for calculating the
	 * center-center distances. They need to be stored so that we can use them
	 * for the optimizations.
	 */
	virtual void update_s(int threadId);

	/* Override how the centers are moved. Before the movement, we need to
	 * backup the old location of centroids and after movement we need to
	 * calculate the tigher upper bound update and the neighbors set.
	 */
	virtual void move_centers(int threadId);

	/* Calculates the lower bound update assuming that all the cached values
	 * are properly calculated. This, default implementation assumes a single
	 * lower bound and therefore takes max update over centroids that fulfil
	 * the neighbor condition. This is true for Hamerly's algorithm, the heap
	 * algorithm and the annular algorithm. In the case of Elkan's algorithm
	 * this method is implemented differently.
	 */
	virtual void calculate_lower_bound_update(int threadId);

	/* Updates the cached inner products. We cache for each centroid its new
	 * norm, its old norm and also the inner product of the old centroid location
	 * and the new centroid location.
	 */
	void update_cached_inner_products(int threadId);

	/* Calculates the maximum upper bound by simple iteration over upper
	 * bound array.
	 */
	virtual void calculate_max_upper_bound(int threadId);

	#ifdef USE_THREADS
	/* This method is used in multithreaded version to collect the maximum upper
	 * bound from the results that are produced by each thread. Each thread outputs
	 * the maximum upper bound per cluster for points that it manages into the
	 * maxUpperBoundAgg array. This method converts the result into maximumUpperBound
	 * array. All threads must be synchronized before calling this method. */
	void aggregate_maximum_upper_bound(int threadId);
	#endif

	/* Comarator of two points by their movement. The points that have moved
	 * more will be first if the standard order is used.
	 *
	 * Parameters: The indices of centroids that have moved to compare.
	 */
	bool center_movement_comparator_function(int c1, int c2);

	/**
	 * Calculates the tigher update.
	 *
	 * Parameters:
	 *  C the center whose update should be calculated
	 *  c the center that moved
	 *  consider_negative Should the calculation consider the negative value.
	 *       This option should be true for Elkan kmeans. Negative update calculation
	 *       costs a bit more, but gives tighter update.
	 * Returns: tigher update of the lower bound of points assigned to C if we
	 *       consider only centroid c
	 */
	double calculate_update(const unsigned int C, const unsigned int c, bool consider_negative = false);

	/*
	 * Calculates the neighbors of centroid C and fills the corresponding row
	 * of the neighbors array. The neighbor array contains for each cluster
	 * a list of centroids that can be second closest to any point of the cluster.
	 * In this implementation we exclude from the list the cluster centroid.
	 *
	 * Parameters:
	 *  C the centroid for that we need to know the neighbors
	 * Returns: nothing
	 */
	void calculate_neighbors(const int C);

	/* Backup here the old location of the centers. */
	Dataset *oldCenters;

	/* Array for the lower bound update. Here we will store the more tighter
	 * lower bound update.
	 *
	 * Size is set automatically using the numLowerBounds * k.
	 * When numLowerBounds is zero (the heap algorithm), then this has size of k.
	 */
	double *lowerBoundUpdate;

	/* Array for storing maximum upper bound per cluster. */
	double *maxUpperBound;

	/* Temporary array used for finding the maximal upper bound using multiple threads. */
	double *maxUpperBoundAgg;

	/* Norm of the centroids in the last iteration. */
	double *oldCentroidsNorm2;

	/* Norm of the current centroids. */
	double *centroidsNorm2;

	/* Inner product of each centroid's new location with old location. */
	double *oldNewCentroidInnerProduct;

	/* Keep track of the distance (divided by 2) between each pair of
	 * points. */
	double *centerCenterDistDiv2;

	/* Here we will store centroids sorted by how much they moved. This is used for
	 * more effective calculation of the tighter lower bound update so that we can
	 * skip some of the points from the iteration. */
	std::vector<int> centersByMovement;

	/*
	 * Here we will store the set of neighbors. It is k array of k arrays, which
	 * contains the set of neighbors. For each centroid there is list of neighbors,
	 * that ends by -1, which means end.
	 *
	 * As this is needed for evaluating the tighter lower bound update, this array
	 * is filled every time and it is algorithm's responsibility to use it or not.
	 */
	int** neighbours;
};

#endif
