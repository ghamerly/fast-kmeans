/* Authors: Greg Hamerly and Jonathan Drake
 * Feedback: hamerly@cs.baylor.edu
 * See: http://cs.baylor.edu/~hamerly/software/kmeans.php
 * Copyright 2014
 */

/* This file is a common driver for running k-means implementations.
 *
 * The program's behavior is determined by a list of commands read from
 * standard input. Input lines should take one of the following forms:
 *
 * dataset some_dataset.txt
 * initialize k [random|kpp]
 * lloyd
 * hamerly
 * elkan
 * adaptive
 * annulus
 * compare
 * sort
 *
 * There are a number of shorthand alternatives,
 * e.g. init for initialize, data for dataset
 *
 * The dataset of n floating-point values in d-dimensional space is
 * read from the indicated file name and should have the form:
 *
 * n d
 * x(1,1) x(1,2) ... x(1, d)
 * .
 * .
 * .
 * x(n,1) x(n,2) ... x(n, d)
 *
 * Results go to standard output, and some extra
 * status information is also sent to standard error.
 */

#include "dataset.h"
#include "general_functions.h"
#include "stats_functions.h"
#include "hamerly_kmeans.h"
#include "hamerly_kmeans_modified.h"
#include "hamerly_kmeans_neighbors.h"
#include "hamerly_kmeans_neighbors1st.h"
#include "hamerly_kmeans_neighbors_only.h"
#include "annulus_kmeans.h"
#include "annulus_kmeans_modified.h"
#include "drake_kmeans.h"
#include "naive_kmeans.h"
#include "elkan_kmeans.h"
#include "compare_kmeans.h"
#include "sort_kmeans.h"
#include "heap_kmeans.h"
#include "heap_kmeans_modified.h"
#include "heap_kmeans_ubarr.h"
#include "heap_kmeans_ubarr_neighbors.h"
#include "naive_kernel_kmeans.h"
#include "elkan_kernel_kmeans.h"
#include "elkan_kmeans_modified.h"
#include "elkan_kmeans_neighbors.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <string>
#include <map>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdlib>

eval_result* execute(std::string command, Kmeans *algorithm, Dataset const *x, unsigned short k, unsigned short const *assignment,
        int xcNdx, int numThreads, int maxIterations,
        std::vector<int> *numItersHistory
        #ifdef MONITOR_ACCURACY
        , std::vector<double> *sseHistory
        #endif
        );

int main(int argc, char **argv) {
    // The set of data points; the set of centers
    Dataset *x = NULL;
    unsigned short *assignment = NULL;
    unsigned short k;

    // The algorithm being used
    Kmeans *algorithm = NULL;

    #ifdef MONITOR_ACCURACY
    std::vector<double> sseHistory;
    #endif
    std::vector<int> numItersHistory;
    int xcNdx = 0;

    int numThreads = 1;
    int maxIterations = std::numeric_limits<int>::max();
	int repeats = 1;

    // Print header row
    print_header_row();

    // Read the command file
    for (std::string command; std::cin >> command; ) {
        if (command == "threads") {
            std::cin >> numThreads;
            #ifndef USE_THREADS
            if (numThreads > 1) {
                std::cerr << "using only one thread because multithreading is disabled" << std::endl;
            }
            numThreads = 1;
            #endif
        } else if (command == "maxiterations") {
            std::cin >> maxIterations;
            if (maxIterations < 0) {
                maxIterations = std::numeric_limits<int>::max();
            }
        } else if (command == "repeats") {
            std::cin >> repeats;
        } else if (command == "dataset" || command == "data") {
            xcNdx++;

            // Get the file name
            std::string dataFileName;
            std::cin >> dataFileName;

            // Open the data file
            std::ifstream input(dataFileName.c_str());
            if (! input) {
                std::cerr << "Unable to open data file: " << dataFileName << std::endl;
                continue;
            }

            // Read the parameters
            int n, d;
            input >> n >> d;

            // Allocate storage
            delete x;
            delete [] assignment;
            assignment = NULL;
            x = new Dataset(n, d);

            // Read the data values directly into the dataset
            for (int i = 0; i < n * d; ++i) {
                input >> x->data[i];
            }

            // Clean up and print success message
            std::cout << "loaded dataset " << dataFileName << ": n = " << n << ", d = " << d << std::endl;
        } else if (command == "initialize" || command == "init") {
            xcNdx++;
            if (x == NULL) {
                std::cerr << "Please load a dataset first" << std::endl;
                continue;
            }
            // Read in the number of means and the initialization method
            std::string method;
            std::cin >> k >> method;

            // Determine the chosen method
            Dataset *c = NULL;
            if (method == "kpp" || method == "kmeansplusplus") {
                // Perform k-means++ initialization
                std::cout << "initializing with kmeans++: k = " << k << std::endl;
                c = init_centers_kmeanspp(*x, k);
            } else if (method == "random") {
                // Perform random initialization
                std::cout << "initializing with random:  k = " << k << std::endl;
                c = init_centers(*x, k);
            }

            if (! c) {
                std::cerr << "Unrecognized initialization method: " << method << std::endl;
                continue;
            }

            delete [] assignment;
            assignment = new unsigned short[x->n];
            for (int i = 0; i < x->n; ++i) {
                assignment[i] = 0;
            }
            assign(*x, *c, assignment);
            delete c;
        } else if (command == "seed") {
            // Read the random seed
            int seed;
            std::cin >> seed;
            srand(seed);
        } else if (command == "lloyd" || command == "naive") {
            algorithm = new NaiveKmeans();
        } else if (command == "hamerly") {
            algorithm = new HamerlyKmeans();
        } else if (command == "hamerlymodified") {
			algorithm = new HamerlyKmeansModified();
		} else if (command == "hamerlyneighbors") {
			algorithm = new HamerlyKmeansNeighbors();
		} else if (command == "hamerlyneighbors1st") {
            algorithm = new HamerlyKmeansNeighbors1st();
        } else if (command == "hamerlyneighborsonly") {
            algorithm = new HamerlyKmeansNeighborsOnly();
        } else if (command == "annulus" || command == "norm") {
            algorithm = new AnnulusKmeans();
        } else if (command == "annulusmodified") {
            algorithm = new AnnulusKmeansModified();
        } else if (command == "elkan") {
            algorithm = new ElkanKmeans();
        } else if (command == "elkanmodified") {
			algorithm = new ElkanKmeansModified();
		} else if (command == "elkanneighbors") {
            algorithm = new ElkanKmeansNeighbors();
        } else if (command == "drake") {
            // Read the number of bounds
            int b;
            std::cin >> b;

            // Ensure b is in the interval [2, k)
            if (b < 2 || b >= k) {
                std::cerr << "Invalid number of lower bounds: " << b << std::endl;
                continue;
            }

            algorithm = new DrakeKmeans(b);
        } else if (command == "adaptive") {
            // Start at k/4
            int b = k / 4;

            // Fix any degenerate cases
            if (b < 2) b = 2;
            if (k <= b) b = k - 1;

            algorithm = new DrakeKmeans(b);
        } else if (command == "compare") {
            algorithm = new CompareKmeans();
        } else if (command == "sort") {
            algorithm = new SortKmeans();
        } else if (command == "heap") {
            algorithm = new HeapKmeans();
        } else if (command == "heapmodified") {
			algorithm = new HeapKmeansModified();
        } else if (command == "heapubarr") {
            algorithm = new HeapKmeansUBarr();
		} else if (command == "heapubarrneighbors") {
            algorithm = new HeapKmeansUBarrNeighbors();
        } else if (command == "kernel" || command == "elkan_kernel") {
            std::string kernelType;
            std::cin >> kernelType;
            Kernel const *kernel = NULL;
            if (kernelType == "gaussian") {
                double tau;
                std::cin >> tau;
                kernel = new GaussianKernel(tau);
            } else if (kernelType == "linear") {
                kernel = new LinearKernel;
            } else if (kernelType == "polynomial") {
                double add, power;
                std::cin >> add >> power;
                kernel = new PolynomialKernel(add, power);
            }
            if (! kernel) {
                std::cerr << "Invalid kernel specification" << std::endl;
                continue;
            }
            if (command == "kernel") {
                algorithm = new NaiveKernelKmeans(kernel);
            } else if (command == "elkan_kernel") {
                algorithm = new ElkanKernelKmeans(kernel);
            } else {
                delete kernel;
                continue;
            }
        } else if (command == "center") {
            std::cout << "centering dataset" << std::endl;
            centerDataset(x);
        } else if (command == "quit" || command == "exit") {
            break;
        } else {
            std::cerr << "Unrecognized command: <" << command << ">." << std::endl;
        }

        if (algorithm) {
            if (repeats <= 1) {
                execute(command, algorithm, x, k, assignment, xcNdx, numThreads, maxIterations, &numItersHistory
                        #ifdef MONITOR_ACCURACY
                        , &sseHistory
                        #endif
                        );
            } else {
                std::vector<eval_result*> results;
                bool valid = true;
                for (int i = 0; i < repeats; ++i) {
                    eval_result* result = execute(command, algorithm, x, k, assignment, xcNdx, numThreads, maxIterations, &numItersHistory
                            #ifdef MONITOR_ACCURACY
                            , &sseHistory
                            #endif
                            );
                    if (result == NULL) {
                        valid = false;
                        break;
                    }
					results.push_back(result);
				}
				if(valid) {
					print_summary(&results);
				}
				for(unsigned i = 0; i < results.size(); ++i)
					delete results.at(i);
			}
            delete algorithm;
            algorithm = NULL;
        }
    }

    delete x;
    delete [] assignment;

    return 0;
}


rusage get_time() {
    rusage now;
    getrusage(RUSAGE_SELF, &now);
    return now;
}


double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int timeval_subtract(timeval *result, timeval *x, timeval *y) {
    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec) {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000) {
        int nsec = (x->tv_usec - y->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.  tv_usec is certainly positive. */
    result->tv_sec = x->tv_sec - y->tv_sec;
    result->tv_usec = x->tv_usec - y->tv_usec;

    /* Return 1 if result
     * is negative. */
    return x->tv_sec < y->tv_sec;
}

double elapsed_time(rusage *start) {
    rusage now;
    timeval diff;
    getrusage(RUSAGE_SELF, &now);
    timeval_subtract(&diff, &now.ru_utime, &start->ru_utime);

    return (double)diff.tv_sec + (double)diff.tv_usec / 1e6;
}

eval_result* execute(std::string command, Kmeans *algorithm, Dataset const *x, unsigned short k, unsigned short const *assignment,
                     int xcNdx,
                     int numThreads,
                     int maxIterations,
                     std::vector<int> *numItersHistory
        #ifdef MONITOR_ACCURACY
        , std::vector<double> *sseHistory
        #endif
        ) {
    // Check for missing initialization
    if (assignment == NULL) {
        std::cerr << "initialize centers first!" << std::endl;
        return NULL;
    }
    if (x == NULL) {
        std::cerr << "load a dataset first!\n" << std::endl;
        return NULL;
    }

    #ifdef COUNT_DISTANCES
    {
        if (numThreads > 1) {
            std::cerr << "Warning: distance counts are not thread-safe and are likely to be incorrect when using multiple threads." << std::endl;
        }
    }
    #endif

    // Begin executing algorithm
    std::cout << std::setw(35) << algorithm->getName() << "\t" << std::flush;

    // Make a working copy of the set of centers
    unsigned short *workingAssignment = new unsigned short[x->n];
    std::copy(assignment, assignment + x->n, workingAssignment);

    // here we will store the result of the run
    eval_result* run_result = new eval_result();
    run_result->num_threads = numThreads;

    // Time the execution and get the number of iterations
    rusage start_clustering_time = get_time();
    double start_clustering_wall_time = get_wall_time();
    algorithm->initialize(x, k, workingAssignment, numThreads);
    run_result->iterations = algorithm->run(maxIterations);
    #ifdef COUNT_DISTANCES
    run_result->num_distances = algorithm->numDistances;
    run_result->inner_products = algorithm->numInnerProducts;
    run_result->assignment_changes = algorithm->assignmentChanges;
    run_result->bounds_updates = algorithm->boundsUpdates;
    #endif
    run_result->cluster_time = elapsed_time(&start_clustering_time);
    run_result->cluster_wall_time = get_wall_time() - start_clustering_wall_time;
    run_result->memory_usage = getMemoryUsage() / 1024.0;

    #ifdef MONITOR_ACCURACY
    {
        run_result->sse = algorithm->getSSE();

        // verification that we get the same distortion with different algorithms
        while (sseHistory->size() <= (size_t) xcNdx) {
            sseHistory->push_back(sse);
        }
        if (sse != sseHistory->back()) {
            std::cerr << "ERROR: sse = " << sse << " but last SSE was " << sseHistory->back() << std::endl;
        }
    }
    #endif

    // output the result
    print_result(run_result);

    // verification that we get the same number of iterations with different algorithms
    while (numItersHistory->size() <= (size_t) xcNdx) {
        numItersHistory->push_back(run_result->iterations);
    }
    if (run_result->iterations != numItersHistory->back()) {
        std::cerr << "ERROR: iterations = " << run_result->iterations << " but last iterations was " << numItersHistory->back() << std::endl;
    }

    std::cout << std::endl;

    #ifdef ITERATION_STATS
    print_iteration_stats(&(algorithm->stats));
    #endif

    delete [] workingAssignment;
    return run_result;
}

