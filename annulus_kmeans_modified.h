#ifndef ANNULUS_KMEANS_MODIFIED_H
#define ANNULUS_KMEANS_MODIFIED_H

/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * This version of Annulus algorithm implements the tighter update.
 */

#include "hamerly_kmeans_modified.h"
#include <utility>

class AnnulusKmeansModified;

class AnnulusKmeansModified : public HamerlyKmeansModified {
    public:
        AnnulusKmeansModified() : xNorm(NULL), cOrder(NULL), guard(NULL) {}
        virtual ~AnnulusKmeansModified() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "annulusmodified"; }

    protected:
        virtual int runThread(int threadId, int maxIterations);
        void sort_means_by_norm();

        // The norm of each point.
        double *xNorm;

        // The order of the centers (first is the norm, second is the center
        // index).
        std::pair<double, int> *cOrder;

        // Guard is an index, for each point, of what is (or might be) the
        // second-closest center. When we tighten the bound, it is the
        // second-closest; however, this is not guaranteed to hold as the bounds
        // change. It is still useful as an index of a close center that is not
        // the closest.
        unsigned short *guard;
};

#endif
