#include "stats_functions.h"
#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstring>

void stdev(eval_result* target, eval_result* sum, eval_result* sum_of_squares, int n);
double stdev(int n, double sum_of_squares, double sum);
eval_result* get_min(std::vector<eval_result*> *data);
eval_result* get_max(std::vector<eval_result*> *data);
eval_result* get_average(std::vector<eval_result*> *data);
eval_result* get_stdev(std::vector<eval_result*> *data);

void print_result(eval_result* result) {
    // Report results
    std::cout << std::setw(5) << result->iterations << "\t";
    std::cout << std::setw(10) << result->num_threads << "\t";
    std::cout << std::setw(10) << result->cluster_time << "\t";
    std::cout << std::setw(10) << result->cluster_wall_time << "\t";
    std::cout << std::setw(8) << result->memory_usage; // from kilo to mega
    #ifdef MONITOR_ACCURACY
    std::cout << "\t" << std::setw(8) << result->sse;
    #endif
    #ifdef COUNT_DISTANCES
    std::cout << "\t" << std::setw(11) << result->num_distances;
    std::cout << "\t" << std::setw(11) << result->inner_products;
    std::cout << "\t" << std::setw(11) << result->assignment_changes;
    std::cout << "\t" << std::setw(11) << result->bounds_updates;
    #endif
}

void print_summary(std::vector<eval_result*>* results) {
    std::cout << "---------------------------------------" << std::endl;
    std::cout << std::setw(35) << "min\t";
    eval_result* min = get_min(results);
    print_result(min);
    delete min;
    std::cout << std::endl << std::setw(35) << "max\t";
    eval_result* max = get_max(results);
    print_result(max);
    delete max;
    std::cout << std::endl << std::setw(35) << "average\t";
    eval_result* avg = get_average(results);
    print_result(avg);
    delete avg;
    std::cout << std::endl << std::setw(35) << "standard deviation\t";
    eval_result* stdev = get_stdev(results);
    print_result(stdev);
    delete stdev;
    std::cout << std::endl << "---------------------------------------" << std::endl;
}

void print_header_row() {
    std::cout << std::setw(35) << "algorithm" << "\t"
            << std::setw(5) << "iters" << "\t"
            << std::setw(10) << "numThreads" << "\t"
            << std::setw(10) << "cpu_secs" << "\t"
            << std::setw(10) << "wall_secs" << "\t"
            << std::setw(8) << "MB"
            #ifdef MONITOR_ACCURACY
            << "\t" << std::setw(8) << "sse"
            #endif
            #ifdef COUNT_DISTANCES
            << "\t" << std::setw(11) << "#distances"
            << "\t" << std::setw(11) << "#i. products"
            << "\t" << std::setw(11) << "#assig. changes"
            << "\t" << std::setw(11) << "#boundsUpdts."
            #endif
            << std::endl;
}

void stdev(eval_result* target, eval_result* sum, eval_result* sum_of_squares, int n) {
    target->iterations = stdev(n, sum_of_squares->iterations, sum->iterations);
    target->num_threads = stdev(n, sum_of_squares->num_threads, sum->num_threads);
    target->cluster_time = stdev(n, sum_of_squares->cluster_time, sum->cluster_time);
    target->cluster_wall_time = stdev(n, sum_of_squares->cluster_wall_time, sum->cluster_wall_time);
    target->memory_usage = stdev(n, sum_of_squares->memory_usage, sum->memory_usage);
    #ifdef MONITOR_ACCURACY
    target->sse = stdev(n, sum_of_squares->sse, sum->sse);
    #endif
    #ifdef COUNT_DISTANCES
    target->num_distances = stdev(n, sum_of_squares->num_distances, sum->num_distances);
    target->inner_products = stdev(n, sum_of_squares->inner_products, sum->inner_products);
    target->assignment_changes = stdev(n, sum_of_squares->assignment_changes, sum->assignment_changes);
    target->bounds_updates = stdev(n, sum_of_squares->bounds_updates, sum->bounds_updates);
    #endif
}

double stdev(int n, double sum_of_squares, double sum) {
    return sqrt(1.0 / n * sum_of_squares - 1.0 / n / n * sum * sum);
}

eval_result* get_min(std::vector<eval_result*> *data) {
    eval_result* retval = new eval_result();
    std::memcpy(retval, data->at(0), sizeof(eval_result));
    for (unsigned int i = 1; i < data->size(); ++i) {
        eval_result* result = data->at(i);
        retval->min(result);
    }
    return retval;
}

eval_result* get_max(std::vector<eval_result*> *data) {
    eval_result* retval = new eval_result();
    std::memcpy(retval, data->at(0), sizeof(eval_result));
    for (unsigned int i = 1; i < data->size(); ++i) {
        eval_result* result = data->at(i);
        retval->max(result);
    }
    return retval;
}

eval_result* get_average(std::vector<eval_result*> *data) {
    eval_result* retval = new eval_result();
    const unsigned int count = data->size();
    for (unsigned int i = 0; i < count; ++i) {
        eval_result* result = data->at(i);
        retval->add(result);
    }

    retval->divide(count);
    return retval;
}

eval_result* get_stdev(std::vector<eval_result*> *data) {
    eval_result* retval = new eval_result();
    eval_result* sum = new eval_result();
    eval_result* sumSq = new eval_result();
    const unsigned int count = data->size();
    for (unsigned int i = 0; i < count; ++i) {
        eval_result* result = data->at(i);
        sum->add(result);
        sumSq->add_square(result);
    }

    stdev(retval, sum, sumSq, count);

    delete sum;
    delete sumSq;
    return retval;
}