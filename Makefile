# compile up to 8 things at once
MAKEFLAGS = -j 8

# Compiler options
CPPFLAGS = -Wall -Werror -pedantic
CPPFLAGS += -g
CPPFLAGS += -O3
CPPFLAGS += -Wno-long-long

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

SRC = $(shell ls *.cpp)

OBJS = $(SRC:.cpp=.o)

all: kmeans

kmeans: $(OBJS)
	g++ $(OBJS) $(CPPFLAGS) $(LDFLAGS) -o kmeans

.PHONY: clean all profile

profile:
	gprof kmeans | less

clean:
	rm -f kmeans $(OBJS) gmon.out
