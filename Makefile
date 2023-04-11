CFLAGS += -DCL_TARGET_OPENCL_VERSION=300
CFLAGS += -Wno-deprecated-declarations
CFLAGS += -fopenmp

main: src/main.c src/helpers.c src/lodepng.c
	gcc -I inc src/*.c -lOpenCL -o main $(CFLAGS)

clean:
	rm -f main debug ./crosscheck.png ./occlusionfill.png ./resized_left.png ./resized_right.png

debug: src/$(mainfile).c src/helpers.c src/lodepng.c 
	clang -I inc src/$(mainfile).c src/helpers.c src/lodepng.c -lOpenCL -o debug -D_FORTIFY_SOURCE=2 -lm  -g3 -O0 -Wall -Wextra -fsanitize=leak -fsanitize=address $(CFLAGS)

build: src/$(mainfile).c src/helpers.c src/lodepng.c 
	gcc -I inc src/$(mainfile).c src/helpers.c src/lodepng.c -lOpenCL -lm -o main $(CFLAGS)
