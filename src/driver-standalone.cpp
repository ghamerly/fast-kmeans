#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include "general_functions.h"
#include "kmeans.h"

#include "annulus_kmeans.h"
#include "compare_kmeans.h"
#include "dataset.h"
#include "elkan_kmeans.h"
#include "general_functions.h"
#include "hamerly_kmeans.h"
#include "heap_kmeans.h"
#include "naive_kmeans.h"
#include "sort_kmeans.h"

Dataset *load_dataset(std::string const &filename) {
    std::ifstream input(filename.c_str());

    int n, d;
    input >> n >> d;
    Dataset *x = new Dataset(n, d);

    for (int i = 0; i < n * d; ++i) input >> x->data[i];

    return x;
}

Kmeans *get_algorithm(std::string const &name) {
    if (name == "kmeans" || name == "lloyd" || name == "naive") return new NaiveKmeans();
    if (name == "hamerly") return new HamerlyKmeans();
    if (name == "annulus") return new AnnulusKmeans();
    if (name == "elkan") return new ElkanKmeans();
    if (name == "compare") return new CompareKmeans();
    if (name == "sort") return new SortKmeans();
    if (name == "heap") return new HeapKmeans();
    assert(false);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " algorithm dataset k [centers|assignment]\n";
        return 1;
    }

    std::string algorithm_name(argv[1]);
    std::string filename(argv[2]);
    int k = std::stoi(argv[3]);
    std::string output(argv[4]);

    Dataset *x = load_dataset(filename);
    Kmeans *algorithm = get_algorithm(algorithm_name);

    Dataset *initialCenters = init_centers_kmeanspp_v2(*x, k);

    unsigned short *assignment = new unsigned short[x->n];

    assign(*x, *initialCenters, assignment);

    algorithm->initialize(x, k, assignment, 1);

    algorithm->run(10000);

    Dataset const *finalCenters = algorithm->getCenters();
    if (output == "centers") {
        finalCenters->print();
    } else {
        assign(*x, *finalCenters, assignment);
        for (int i = 0; i < x->n; ++i) {
            std::cout << assignment[i] << "\n";
        }
    }

    delete x;
    delete algorithm;
    delete initialCenters;
    delete [] assignment;

    return 0;
}
