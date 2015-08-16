/*
 * File:   stats_functions.h
 * Author: Petr Rysavy
 *
 * Created on 16. August 2015, 12:57
 */

#ifndef STATS_FUNCTIONS_H
#define	STATS_FUNCTIONS_H

#include <vector>

/* The result of execution of algorithm. */
struct eval_result {
    // number of iterations that were needed to reach the optimum
    int iterations;
    // number of threads used
    int num_threads;
    // CPU time needed for calculations
    double cluster_time;
    // wall time needed for calculations
    double cluster_wall_time;
    // memory usage
    double memory_usage;
    #ifdef MONITOR_ACCURACY
    double sse;
    #endif
    #ifdef COUNT_DISTANCES
    // number of distances evaluated by the algorithm
    long long num_distances;
    // number of inner products evaluated by the algorithm; does not include num_distances
    long long inner_products;
    // how many times did a point change its assignment
    long long assignment_changes;
    // total update of the bounds
    double bounds_updates;
    #endif

    void add(const eval_result* a) {
        iterations += a->iterations;
        num_threads += a->num_threads;
        cluster_time += a->cluster_time;
        cluster_wall_time += a->cluster_wall_time;
        memory_usage += a->memory_usage;
        #ifdef MONITOR_ACCURACY
        sse += a->sse;
        #endif
        #ifdef COUNT_DISTANCES
        num_distances += a->num_distances;
        inner_products += a->inner_products;
        assignment_changes += a->assignment_changes;
        bounds_updates += a->bounds_updates;
        #endif
    }

    void divide(const int n) {
        iterations = ((double) iterations) / n;
        num_threads /= n;
        cluster_time /= n;
        cluster_wall_time /= n;
        memory_usage /= n;
        #ifdef MONITOR_ACCURACY
        sse /= n;
        #endif
        #ifdef COUNT_DISTANCES
        num_distances /= n;
        inner_products /= n;
        assignment_changes /= n;
        bounds_updates /= n;
        #endif
    }

    void min(const eval_result* a) {
        iterations = std::min(iterations, a->iterations);
        num_threads = std::min(num_threads, a->num_threads);
        cluster_time = std::min(cluster_time, a->cluster_time);
        cluster_wall_time = std::min(cluster_wall_time, a->cluster_wall_time);
        memory_usage = std::min(memory_usage, a->memory_usage);
        #ifdef MONITOR_ACCURACY
        sse = std::min(sse, a->sse);
        #endif
        #ifdef COUNT_DISTANCES
        num_distances = std::min(num_distances, a->num_distances);
        inner_products = std::min(inner_products, a->inner_products);
        assignment_changes = std::min(assignment_changes, a->assignment_changes);
        bounds_updates = std::min(bounds_updates, a->bounds_updates);
        #endif
    }

    void max(const eval_result* a) {
        iterations = std::max(iterations, a->iterations);
        num_threads = std::max(num_threads, a->num_threads);
        cluster_time = std::max(cluster_time, a->cluster_time);
        cluster_wall_time = std::max(cluster_wall_time, a->cluster_wall_time);
        memory_usage = std::max(memory_usage, a->memory_usage);
        #ifdef MONITOR_ACCURACY
        sse = std::max(sse, a->sse);
        #endif
        #ifdef COUNT_DISTANCES
        num_distances = std::max(num_distances, a->num_distances);
        inner_products = std::max(inner_products, a->inner_products);
        assignment_changes = std::max(assignment_changes, a->assignment_changes);
        bounds_updates = std::max(bounds_updates, a->bounds_updates);
        #endif
    }

    void add_square(const eval_result* a) {
        iterations += a->iterations * a->iterations;
        num_threads += a->num_threads * a->num_threads;
        cluster_time += a->cluster_time * a->cluster_time;
        cluster_wall_time += a->cluster_wall_time * a->cluster_wall_time;
        memory_usage += a->memory_usage * a->memory_usage;
        #ifdef MONITOR_ACCURACY
        sse += a->sse * a->sse;
        #endif
        #ifdef COUNT_DISTANCES
        num_distances += a->num_distances * a->num_distances;
        inner_products += a->inner_products * a->inner_products;
        assignment_changes += a->assignment_changes * a->assignment_changes;
        bounds_updates += a->bounds_updates * a->bounds_updates;
        #endif
    }
};

void print_result(eval_result* result);
void print_summary(std::vector<eval_result*>* results);
void print_header_row();

#endif	/* STATS_FUNCTIONS_H */
