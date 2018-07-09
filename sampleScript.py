#!/usr/bin/env python3

### Imports ###

import random

from fastkmeans import *

### Main ###

# Test variables

k = 3
drake_lower_bounds = 2
assert drake_lower_bounds < k
maxiterations = 100

# Read in sample data

with open('../smallDataset.txt') as f:
    text = f.readlines()
    n, d = map(int, text[0].strip().split())
    x = Dataset(n, d)

    for i, line in enumerate(text[1:]):
        vals = map(float, line.strip().split())
        for j, v in enumerate(vals):
            x[i,j] = v

print('size of dataset:', n)
print('dimensions:', d)

# Populate assignments

initial_ctrs = kmeans_plusplus(x, k)
print('\ninitial centers (kmeans++):\n')
initial_ctrs.print()

a = Assignment(n)
assign(x, initial_ctrs, a)

# Initialize and run each algorithm

algorithms = (
    Annulus,
    Compare,
    Drake,
    Elkan,
    Hamerly,
    Heap,
    Naive,
    Sort,
)

for alg in algorithms:
    algorithm = alg() if alg is not Drake else alg(drake_lower_bounds)
    print('\n{}'.format(algorithm.get_name()))
    algorithm.initialize(x, k, a)
    algorithm.run(maxiterations)

    centers = algorithm.centers
    print('\ncenters:\n')
    centers.print()
    print()
