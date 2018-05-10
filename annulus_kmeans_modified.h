#ifndef ANNULUS_KMEANS_MODIFIED_H
#define ANNULUS_KMEANS_MODIFIED_H

/* Authors: Greg Hamerly and Jonathan Drake and Petr Ryšavý
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2015
 *
 * This version of Annulus algorithm implements the tighter update. Most of the
 * code is copied from the default version of Annulus algorithm, only the parent
 * class is HamerlyKmeansModified.
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

        double *xNorm;

        std::pair<double, int> *cOrder;

        unsigned short *guard;
};

#endif
