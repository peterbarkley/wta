OPT=-O3
CFLAGS=-Wall -Wextra -Wwrite-strings -pedantic -funroll-loops -Wstrict-prototypes -std=c99 -march=native -mtune=native
SRC_FILES = $(wildcard src/c/*.c)

simplex_proj: $(SRC_FILES)
	mkdir -p ./build
	$(CC) -Wall -fPIC -shared -o ./build/$@.so $(OPT) $(SRC_FILES) $(CFLAGS) 

.PHONY: clean

clean:
	rm -rf ./build
