# Compile and link flags
CXX          =  g++
CFLAGS       = -Wall -g

# Compilation (add flags as needed)
CXXFLAGS    += `pkg-config opencv --cflags` -Wall -g

# Linking (add flags as needed)
LDFLAGS     += `pkg-config opencv --libs`

# Name your target executables here
all         = test

# Default target is the first one - so we will have it make everything :-)
all: $(all)

clean:
	rm -f $(all) *.o

# Program dependencies (.o files will be compiles by implicit rules)
test:  test.cpp DenseOF.o SimpleFlowDenseOF.o
	$(CXX) test.cpp DenseOF.o SimpleFlowDenseOF.o $(CXXFLAGS) $(LDFLAGS) -o test

