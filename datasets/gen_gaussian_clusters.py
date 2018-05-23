import random
import sys

def random_point(center, spread):
    return tuple(random.gauss(ci, spread) for ci in center)

n = int(sys.argv[1])
d = int(sys.argv[2])
k = int(sys.argv[3])
s = int(sys.argv[4])

centers = [random_point([0.0] * d, 1.0) for _ in range(k)]
points = [random_point(random.choice(centers), s) for _ in range(n)]

print n, d
for p in points:
    print ' '.join(map(lambda v: '{0: 0.3f}'.format(v), p))
