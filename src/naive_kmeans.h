#ifndef NAIVE_KMEANS_H
#define NAIVE_KMEANS_H

/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 *
 * NaiveKmeans is the standard k-means algorithm that has no acceleration
 * applied. Also known as Lloyd's algorithm.
 */

#include "original_space_kmeans.h"

class NaiveKmeans : public OriginalSpaceKmeans {
    public:
        virtual std::string getName() const { return "naive"; }
        virtual ~NaiveKmeans() { free(); }
    protected:
        virtual int runThread(int threadId, int maxIterations);
};

#endif

