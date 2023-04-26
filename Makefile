CFLAGS += -DCL_TARGET_OPENCL_VERSION=200
CFLAGS += -Wno-deprecated-declarations
CFLAGS += -fopenmp


main: src/main.c src/helpers.c src/lodepng.c
	gcc -I inc src/*.c -lOpenCL -o main $(CFLAGS)

clean:
	rm -f main debug ./*.png

debug: src/$(mainfile).c src/helpers.c src/lodepng.c 
	clang -I inc src/$(mainfile).c src/helpers.c src/lodepng.c -latomic -std=c11 -lOpenCL -o debug -D_FORTIFY_SOURCE=2 -lm  -g3 -O0 -Wall -Wextra -fsanitize=leak -fsanitize=address $(CFLAGS)

build: src/$(mainfile).c src/helpers.c src/lodepng.c 
	gcc -I inc src/$(mainfile).c src/helpers.c src/lodepng.c -latomic -std=c11 -lOpenCL -lm -g3 -o main $(CFLAGS)

build-clang: src/$(mainfile).c src/helpers.c src/lodepng.c 
	clang -I inc src/$(mainfile).c src/helpers.c src/lodepng.c -latomic -std=c11 -lOpenCL -o main -D_FORTIFY_SOURCE=2 -lm  -g3 -O0 -Wall -Wextra $(CFLAGS)