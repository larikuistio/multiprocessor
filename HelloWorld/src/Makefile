CFLAGS += -DCL_TARGET_OPENCL_VERSION=300
CFLAGS += -Wno-deprecated-declarations

helloworld: helloworld.c hello.cl
	gcc helloworld.c -lOpenCL -o helloworld $(CFLAGS)

clean:
	rm -f helloworld