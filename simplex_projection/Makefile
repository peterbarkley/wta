OPT=-O3
CFLAGS=-Wall -Wextra -Wwrite-strings -pedantic -funroll-loops -Wstrict-prototypes -std=c99 -march=native -mtune=native
OMP=-fopenmp -D PARALLEL=1
SRC_FILES = $(wildcard src/c/*.c)

all: parallel_simplex_proj serial_simplex_proj

parallel_simplex_proj: $(SRC_FILES)
	mkdir -p ./build
	$(CC) -fPIC -shared -o ./build/$@.so $(OPT) $(SRC_FILES) $(CFLAGS) $(OMP)

serial_simplex_proj: $(SRC_FILES)
	mkdir -p ./build
	$(CC) -fPIC -shared -o ./build/$@.so $(OPT) $(SRC_FILES) $(CFLAGS)

.PHONY: clean

clean:
	rm -rf ./build
