# compile up to 8 things at once
MAKEFLAGS = -j 8

# Compiler options
CPPFLAGS = -Wall -Werror -pedantic
CPPFLAGS += -fPIC
CPPFLAGS += -g
CPPFLAGS += -O3
CPPFLAGS += -Wno-long-long
CPPFLAGS += -std=c++11

# Verify algorithm correctness while debugging
#CPPFLAGS += -DDEBUG
#CPPFLAGS += -DVERIFY_ASSIGNMENTS

# To use pthreads, uncomment both lines below.
#CPPFLAGS += -DUSE_THREADS
#LDFLAGS += -lpthread

# Monitor internal algorithm effectiveness
#CPPFLAGS += -DCOUNT_DISTANCES
#CPPFLAGS += -DMONITOR_ACCURACY

# Enable code profiling
#CPPFLAGS += -pg

LIBSRC = $(shell ls *.cpp | grep -v driver)
LIBOBJS = $(LIBSRC:.cpp=.o)

KMEANSLIBRARY = libkmeans.a

all: driver-experiment driver-standalone projection-test

$(KMEANSLIBRARY): $(LIBOBJS)
	ar -cq $(KMEANSLIBRARY) $(LIBOBJS)

driver-experiment: $(KMEANSLIBRARY) driver-experiment.o
	g++ -L . $(CPPFLAGS) $(LDFLAGS) driver-experiment.o -o driver-experiment -lkmeans

driver-standalone: $(KMEANSLIBRARY) driver-standalone.o
	g++ -L . $(CPPFLAGS) $(LDFLAGS) driver-standalone.o -o driver-standalone -lkmeans

projection-test: $(KMEANSLIBRARY) projection-test.o
	g++ -L . $(CPPFLAGS) $(LDFLAGS) projection-test.o -o projection-test -lkmeans

.PHONY: clean all

clean:
	rm -f $(KMEANSLIBRARY) driver-experiment driver-standalone *.o gmon.out
