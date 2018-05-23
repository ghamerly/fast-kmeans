import random
import sys

n = int(sys.argv[1])
d = int(sys.argv[2])

print n, d
for _ in range(n):
    print ' '.join(map(lambda v: '{0: 0.3f}'.format(v), [random.uniform(-10, 10) for __ in range(d)]))
