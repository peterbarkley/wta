OPT=-O3
CFLAGS=-Wall -Wextra -Wwrite-strings -pedantic -funroll-loops -Wstrict-prototypes -std=c99 -march=native -mtune=native
OMP=-fopenmp
PARALLEL_SRC_FILES = $(wildcard src/c/*parallel.c)
SERIAL_SRC_FILES = $(wildcard src/c/*serial.c)

all: parallel_simplex_proj serial_simplex_proj

parallel_simplex_proj: $(PARALLEL_SRC_FILES)
	mkdir -p ./build
	$(CC) -Wall -fPIC -shared -o ./build/$@.so $(OPT) $(PARALLEL_SRC_FILES) $(CFLAGS) $(OMP)

serial_simplex_proj: $(SERIAL_SRC_FILES)
	mkdir -p ./build
	$(CC) -Wall -fPIC -shared -o ./build/$@.so $(OPT) $(SERIAL_SRC_FILES) $(CFLAGS)

.PHONY: clean

clean:
	rm -rf ./build
