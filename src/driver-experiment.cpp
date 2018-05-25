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
#include "hamerly_kmeans.h"
#include "annulus_kmeans.h"
#include "drake_kmeans.h"
#include "naive_kmeans.h"
#include "elkan_kmeans.h"
#include "compare_kmeans.h"
#include "sort_kmeans.h"
#include "heap_kmeans.h"
#include "naive_kernel_kmeans.h"
#include "elkan_kernel_kmeans.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <string>
#include <map>
#include <ctime>
#include <unistd.h>
#include <cstdlib>

void execute(std::string command, Kmeans *algorithm, Dataset const *x, unsigned short k, unsigned short const *assignment,
        unsigned short *outAssignment, Dataset *outCenters,
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
    unsigned short *outAssignment = NULL;
    Dataset *outCenters = NULL;

    // The algorithm being used
    Kmeans *algorithm = NULL;

    #ifdef MONITOR_ACCURACY
    std::vector<double> sseHistory;
    #endif
    std::vector<int> numItersHistory;
    int xcNdx = 0;

    int numThreads = 1;
    int maxIterations = std::numeric_limits<int>::max();

    // Print header row
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
              #endif
              << std::endl;

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
            delete [] outAssignment;
            delete outCenters;
            assignment = NULL;
            outAssignment = NULL;
            outCenters = NULL;
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
                c = init_centers_kmeanspp_v2(*x, k);
            } else if (method == "random") {
                // Perform random initialization
                std::cout << "initializing with random:  k = " << k << std::endl;
                c = init_centers(*x, k);
            }

            delete outCenters;
            outCenters = new Dataset(k, x->d);
            *outCenters = *c;

            if (! c) {
                std::cerr << "Unrecognized initialization method: " << method << std::endl;
                continue;
            }

            delete [] assignment;
            assignment = new unsigned short[x->n];
            outAssignment = new unsigned short[x->n];
            std::fill(assignment, assignment + x->n, 0);
            assign(*x, *c, assignment);
            std::copy(assignment, assignment + x->n, outAssignment);
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
        } else if (command == "annulus" || command == "norm") {
            algorithm = new AnnulusKmeans();
        } else if (command == "elkan") {
            algorithm = new ElkanKmeans();
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
        } else if (command == "dump_assignment") {
            if (outAssignment) {
                for (int i = 0; i < x->n; ++i) {
                    std::cout << outAssignment[i] << std::endl;
                }
            } else {
                std::cerr << "Error: no assignment available" << std::endl;
            }
        } else if (command == "dump_centers") {
            if (outCenters) {
                outCenters->print();
            } else {
                std::cerr << "Error: no centers available" << std::endl;
            }
        } else if (command == "quit" || command == "exit") {
            break;
        } else {
            std::cerr << "Unrecognized command: <" << command << ">." << std::endl;
        }

        if (algorithm) {
            execute(command, algorithm, x, k, assignment, 
                    outAssignment, outCenters,
                    xcNdx, numThreads, maxIterations, &numItersHistory
                    #ifdef MONITOR_ACCURACY
                    , &sseHistory
                    #endif
                   );
            delete algorithm;
            algorithm = NULL;
        }
    }

    delete x;
    delete [] assignment;

    return 0;
}

void execute(std::string command, Kmeans *algorithm, Dataset const *x, unsigned short k, unsigned short const *assignment,
        unsigned short *outAssignment, Dataset *outCenters,
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
        return;
    }
    if (x == NULL) {
        std::cerr << "load a dataset first!\n" << std::endl;
        return;
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
    unsigned short *workingAssignment = outAssignment ? outAssignment : new unsigned short[x->n];
    std::copy(assignment, assignment + x->n, workingAssignment);

    // Time the execution and get the number of iterations
    rusage start_clustering_time = get_time();
    double start_clustering_wall_time = get_wall_time();
    algorithm->initialize(x, k, workingAssignment, numThreads);
    int iterations = algorithm->run(maxIterations);

    if (outCenters) {
        // try to grab the centers, if they exist
        Dataset const *ctrs = algorithm->getCenters();
        if (ctrs)
            *outCenters = *ctrs;
    }

    #ifdef COUNT_DISTANCES
    long long numDistances = algorithm->numDistances;
    #endif
    double cluster_time = elapsed_time(&start_clustering_time);
    double cluster_wall_time = get_wall_time() - start_clustering_wall_time;

    // Report results
    std::cout << std::setw(5) << iterations << "\t";
    std::cout << std::setw(10) << numThreads << "\t";
    std::cout << std::setw(10) << cluster_time << "\t";
    std::cout << std::setw(10) << cluster_wall_time << "\t";
    std::cout << std::setw(8) << (getMemoryUsage() / 1024.0); // from kilo to mega
    #ifdef MONITOR_ACCURACY
    {
        double sse = algorithm->getSSE();
        std::cout << "\t" << std::setw(11) << sse;

        // verification that we get the same distortion with different algorithms
        while (sseHistory->size() <= (size_t)xcNdx) {
            sseHistory->push_back(sse);
        }
        if (sse != sseHistory->back()) {
            std::cerr << "ERROR: sse = " << sse << " but last SSE was " << sseHistory->back() << std::endl;
        }
    }
    #endif
    #ifdef COUNT_DISTANCES
    {
        std::cout << "\t" << std::setw(11) << numDistances;
    }
    #endif

    // verification that we get the same number of iterations with different algorithms
    while (numItersHistory->size() <= (size_t)xcNdx) {
        numItersHistory->push_back(iterations);
    }
    if (iterations != numItersHistory->back()) {
        std::cerr << "ERROR: iterations = " << iterations << " but last iterations was " << numItersHistory->back() << std::endl;
    }

    std::cout << std::endl;

    if (!outAssignment)
        delete [] workingAssignment;
}

