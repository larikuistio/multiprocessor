CFLAGS += -DCL_TARGET_OPENCL_VERSION=300
CFLAGS += -Wno-deprecated-declarations

main: src/main.c src/helpers.c src/lodepng.c 
	gcc src/main.c src/helpers.c src/lodepng.c -lOpenCL -o main $(CFLAGS)

