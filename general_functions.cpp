/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

#include "dataset.h"
#include "kmeans.h"
#include "general_functions.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdio>
#include <unistd.h>

void addVectors(double *a, double const *b, int d) {
    double const *end = a + d;
    while (a < end) {
        *(a++) += *(b++);
    }
}

void subVectors(double *a, double const *b, int d) {
    double const *end = a + d;
    while (a < end) {
        *(a++) -= *(b++);
    }
}

double distance2silent(double const *a, double const *b, int d) {
    double d2 = 0.0, diff;
    double const *end = a + d; // one past the last valid entry in a
    while (a < end) {
        diff = *(a++) - *(b++);
        d2 += diff * diff;
    }
    return d2;
}

void centerDataset(Dataset *x) {
    double *xCentroid = new double[x->d];

    for (int d = 0; d < x->d; ++d) {
        xCentroid[d] = 0.0;
    }

    for (int i = 0; i < x->n; ++i) {
        addVectors(xCentroid, x->data + i * x->d, x->d);
    }

    // compute average (divide by n)
    for (int d = 0; d < x->d; ++d) {
        xCentroid[d] /= x->n;
    }

    // re-center the dataset
    const double *xEnd = x->data + x->n * x->d;
    for (double *xp = x->data; xp != xEnd; xp += x->d) {
        subVectors(xp, xCentroid, x->d);
    }

    delete [] xCentroid;
}

Dataset *init_centers(Dataset const &x, unsigned short k) {
    int *chosen_pts = new int[k];
    Dataset *c = new Dataset(k, x.d);
    for (int i = 0; i < k; ++i) {
        bool acceptable = true;
        do {
            acceptable = true;
            chosen_pts[i] = rand() % x.n;
            for (int j = 0; j < i; ++j) {
                if (chosen_pts[i] == chosen_pts[j]) {
                    acceptable = false;
                    break;
                }
            }
        } while (!acceptable);
        double *cdp = c->data + i * x.d;
        memcpy(cdp, x.data + chosen_pts[i] * x.d, sizeof(double) * x.d);
        if (c->sumDataSquared) {
            c->sumDataSquared[i] = std::inner_product(cdp, cdp + x.d, cdp, 0.0);
        }
    }

    delete [] chosen_pts;

    return c;
}

Dataset *init_centers_kmeanspp(Dataset const &x, unsigned short k) {
    int *chosen_pts = new int[k];
    double *dist2 = new double[x.n];
    int *closest = new int[x.n]; // index of centroid that is closest to a point
    // distances between new centroid and all others divided by 2
    double *centroid_dist2_div4 = new double[k - 1];

    // initialize dist2
    std::fill(dist2, dist2 + x.n, std::numeric_limits<double>::max());
    std::fill(closest, closest + x.n, 0);
    std::fill(centroid_dist2_div4, centroid_dist2_div4 + k - 1, 0.0);

    // choose the first point randomly
    int ndx = 1;
    chosen_pts[ndx - 1] = rand() % x.n;

    while (ndx < k) {
        double sum_distribution = 0.0;
        // look for the point that is furthest from any center
        double max_dist = 0.0;
        for (int i = 0; i < x.n; ++i) {
            // we need to consider the new centroid (ndx) as closest to i
            if (dist2[i] > centroid_dist2_div4[closest[i]]) {
                double d2 = distance2silent(x.data + i * x.d, x.data + chosen_pts[ndx - 1] * x.d, x.d);

                if (d2 < dist2[i]) {
                    dist2[i] = d2;
                    closest[i] = ndx - 1;
                }
            }

            if (dist2[i] > max_dist) {
                max_dist = dist2[i];
            }

            sum_distribution += dist2[i];
        }

        bool unique = true;

        do {
            // choose a random interval according to the new distribution
            double r = sum_distribution * (double) rand() / (double) RAND_MAX;
            double sum_cdf = dist2[0];
            int cdf_ndx = 0;
            while (sum_cdf < r) {
                sum_cdf += dist2[++cdf_ndx];
            }
            chosen_pts[ndx] = cdf_ndx;

            for (int i = 0; i < ndx; ++i) {
                unique = unique && (chosen_pts[ndx] != chosen_pts[i]);
            }
        } while (!unique);

        // calculate the squared distance between the new point and all others div 4
        for (int i = 0; i < ndx; ++i) {
            centroid_dist2_div4[i] = distance2silent(x.data + chosen_pts[i] * x.d, x.data + chosen_pts[ndx] * x.d, x.d) / 4.0;
        }

        ++ndx;
    }

    Dataset *c = new Dataset(k, x.d);
    for (int i = 0; i < c->n; ++i) {
        double *cdp = c->data + i * x.d;
        memcpy(cdp, x.data + chosen_pts[i] * x.d, sizeof(double) * x.d);
        if (c->sumDataSquared) {
            c->sumDataSquared[i] = std::inner_product(cdp, cdp + x.d, cdp, 0.0);
        }
    }

    delete [] chosen_pts;
    delete [] dist2;

    return c;
}

/**
 * in MB
 */
double getMemoryUsage() {
    char buf[30];
    snprintf(buf, 30, "/proc/%u/statm", (unsigned) getpid());
    FILE* pf = fopen(buf, "r");
    unsigned int totalProgramSizeInPages = 0;
    unsigned int residentSetSizeInPages = 0;
    if (pf) {
        int numScanned = fscanf(pf, "%u %u" /* %u %u %u %u %u"*/, &totalProgramSizeInPages, &residentSetSizeInPages);
        if (numScanned != 2) {
            return 0.0;
        }
    }

    fclose(pf);
    pf = NULL;

    double sizeInKilobytes = residentSetSizeInPages * 4.0; // assumes 4096 byte page
    // getconf PAGESIZE

    return sizeInKilobytes;
}

void assign(Dataset const &x, Dataset const &c, unsigned short *assignment) {
    for (int i = 0; i < x.n; ++i) {
        double shortestDist2 = std::numeric_limits<double>::max();
        int closest = 0;
        for (int j = 0; j < c.n; ++j) {
            double d2 = 0.0, *a = x.data + i * x.d, *b = c.data + j * x.d;
            for (; a != x.data + (i + 1) * x.d; ++a, ++b) {
                d2 += (*a - *b) * (*a - *b);
            }
            if (d2 < shortestDist2) {
                shortestDist2 = d2;
                closest = j;
            }
        }
        assignment[i] = closest;
    }
}

