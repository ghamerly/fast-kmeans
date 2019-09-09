#ifndef MTI_KMEANS_H
#define MTI_KMEANS_H


#include "triangle_inequality_base_kmeans.h"

class MTIKmeans : public TriangleInequalityBaseKmeans {
    public:
        MTIKmeans() : centerCenterDistDiv2(NULL) {}
        virtual ~MTIKmeans() { free(); }
        virtual void free();
        virtual void initialize(Dataset const *aX, unsigned short aK, unsigned short *initialAssignment, int aNumThreads);
        virtual std::string getName() const { return "mti"; }

    protected:
        virtual int runThread(int threadId, int maxIterations);

        // Update the distances between each pair of centers.
        void update_center_dists(int threadId);

        // Update the upper bounds for the range of points given.
        void update_bounds(int startNdx, int endNdx);

        // Keep track of the distance (divided by 2) between each pair of
        // centers
        double *centerCenterDistDiv2;
};

#endif
