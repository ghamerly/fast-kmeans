/* The purpose of this code is to test the speed and quality of doing the following:
 * 1. load data
 * 2. project to a (much) lower dimension using random projection
 * 3. use a fast low-dimensional clustering algorithm
 * 4. map the centers back to the original space (using cluster assignments as
 *      the mapping to reconstruct the original centers)
 * 5. determine the k-means quality of these centers
 * 6. run a few more iterations of k-means using the original space
 * 7. determine the k-means quality of these centers
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>

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

Dataset const *project_dataset(Dataset const *original, int projected_dimension) {
    if (projected_dimension == 0 || projected_dimension == original->d) {
        return original;
    }

    int d = original->d;

    Dataset *projection = new Dataset(projected_dimension, d); // this is transposed from the normal form to make an easy inner_product below
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0, 1);
    for (int i = 0; i < projection->nd; ++i) {
        projection->data[i] = normal(gen);
    }
    // could normalize for length here...

    Dataset *projected = new Dataset(original->n, projected_dimension);
    for (int i = 0; i < original->n; ++i) {
        auto o = original->data + i * d;
        for (int j = 0; j < projected_dimension; ++j) {
            projected->data[i * projected_dimension + j] = std::inner_product(o, o + d, projection->data + j * d, 0.0);
        }
    }
    delete projection;

    return projected;
}

Dataset const *getOriginalCenters(Dataset const &x, int k, Kmeans const &algorithm) {
    Dataset *c = new Dataset(k, x.d);
    std::fill(c->data, c->data + c->nd, 0.0);

    int *count = new int[k];
    std::fill(count, count + k, 0);

    for (int i = 0; i < x.n; ++i) {
        auto a = algorithm.getAssignment(i);
        assert(0 <= a && a < k);
        addVectors(c->data + a * x.d, x.data + i * x.d, x.d);
        count[a] += 1;
    }

    for (int i = 0; i < k; ++i) {
        if (count[i] == 0)
            continue;
        for (int j = 0; j < x.d; ++j) {
            c->data[i * x.d + j] /= count[i];
        }
    }

    return c;
}

double kmeansObjective(Dataset const &x, Dataset const &centers, Kmeans const &algorithm) {
    double sse = 0.0;
    double *centers_inner_product = new double[centers.n];
    double *data_inner_product = new double[x.n];
    for (int i = 0; i < centers.n; ++i) {
        auto ci = centers.data + i * centers.d;
        centers_inner_product[i] = std::inner_product(ci, ci + centers.d, ci, 0.0);
    }

    for (int i = 0; i < x.n; ++i) {
        auto xi = x.data + i * x.d;
        data_inner_product[i] = std::inner_product(xi, xi + x.d, xi, 0.0);
    }

    // (x - y)'(x - y) = x'x + y'y - 2 * x'y
    for (int i = 0; i < x.n; ++i) {
        auto a = algorithm.getAssignment(i);
        auto xi = x.data + i * x.d;
        sse += centers_inner_product[a]
                + data_inner_product[i]
                - 2 * std::inner_product(xi, xi + x.d, centers.data + a * x.d, 0.0);
    }

    delete [] centers_inner_product;
    delete [] data_inner_product;

    return sse;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "usage: " << argv[0] << " algorithm dataset k projectDim\n";
        return 1;
    }

    std::string algorithm_name(argv[1]);
    std::string filename(argv[2]);
    int k = std::stoi(argv[3]);
    int projectDim = std::stoi(argv[4]);

    auto start_time = get_time();
    Dataset *originalData = load_dataset(filename);
    std::cout << "load dataset time: " << elapsed_time(&start_time) << std::endl;

    Kmeans *algorithm = get_algorithm(algorithm_name);

    start_time = get_time();
    Dataset const *x = project_dataset(originalData, projectDim);
    std::cout << "project dataset time: " << elapsed_time(&start_time) << std::endl;

    start_time = get_time();
    Dataset *initialCenters = init_centers_kmeanspp_v2(*x, k);
    std::cout << "k-means++ time: " << elapsed_time(&start_time) << std::endl;

    unsigned short *assignment = new unsigned short[x->n];
    assign(*x, *initialCenters, assignment);
    algorithm->initialize(x, k, assignment, 1);
    start_time = get_time();
    auto projected_iterations = algorithm->run(10000);
    std::cout << "projected k-means time: " << elapsed_time(&start_time) << " for " << projected_iterations << " iterations\n";

    std::cout << "\noriginalCenters\n";
    Dataset const *originalCenters = getOriginalCenters(*originalData, k, *algorithm);
    std::cout << "k-means objective with only projection " << kmeansObjective(*originalData, *originalCenters, *algorithm) << std::endl;

    // now cluster in the original space
    algorithm->initialize(originalData, k, assignment, 1); // FIXME -- we are abusing the reuse of assignment here
    start_time = get_time();
    auto final_iterations = algorithm->run(20); // Severely limit the number of iterations to save time
    Dataset const *adjustedCenters = getOriginalCenters(*originalData, k, *algorithm);
    std::cout << "original-space k-means time: " << elapsed_time(&start_time) << " for " << final_iterations << " iterations\n";
    std::cout << "k-means objective after adjustment " << kmeansObjective(*originalData, *adjustedCenters, *algorithm) << std::endl;

    if (x != originalData)
        delete x;
    delete originalData;
    delete originalCenters;
    delete algorithm;
    delete initialCenters;
    delete [] assignment;

    return 0;
}
