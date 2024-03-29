// files containing opencl kernels
#define NUM_FILES 7
#define PROGRAM_FILE_0 "kernels/zncc.cl"
#define PROGRAM_FILE_1 "kernels/grayscale.cl"
#define PROGRAM_FILE_2 "kernels/resizeimage.cl"
#define PROGRAM_FILE_3 "kernels/crosscheck.cl"
#define PROGRAM_FILE_4 "kernels/occlusionfill.cl"
#define PROGRAM_FILE_5 "kernels/findminmax.cl"
#define PROGRAM_FILE_6 "kernels/normalize.cl"
// kernel names
#define KERNEL_NAME_0 "resizeimage"
#define KERNEL_NAME_1 "grayscale"
#define KERNEL_NAME_2 "zncc"
#define KERNEL_NAME_3 "crosscheck"
#define KERNEL_NAME_4 "occlusionfill"
#define KERNEL_NAME_5 "findminmax"
#define KERNEL_NAME_6 "normalize"


/*
Some parts are based on examples provided in this book:
	
Matthew Scarpino
OpenCL in action : how to accelerate graphics and computation

https://oula.finna.fi/Record/oy.9917467213906252?sid=2954376879
*/


#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "helpers.h"
#include <limits.h>
#include <sys/time.h>

// ZNCC PARAMETERS
int MIN_DISPARITY = 0;
int MAX_DISPARITY = 65;
int MAX_DISPARITY_NEG = -65;
int B = 8;
// CROSSCHECK PARAMETERS
int THRESHOLD = 2;
// OCCLUSIONFILL PARAMETERS
int NEIGHBORHOOD_SIZE = 256;




int main(int argc, char **argv)
{
	// profile entire program execution time
	clock_t startprogclk = clock();
	double startprog = queryProfiler();

	// exit if required file names not provided
	if (argc < 4) {
		printf("provide 2 input images and one output image filenames as arguments\n");
		return EXIT_FAILURE;
	}

	const char* left = argv[1];
	const char* right = argv[2];
	const char* outputimg = argv[3];
	unsigned char* image_l = 0;
	unsigned char* image_r = 0;
	unsigned char* resized_l = 0;
	unsigned char* resized_r = 0;
	unsigned char* grayscale_l = 0;
	unsigned char* grayscale_r = 0;
	unsigned char* zncc_output_lr = 0;
	unsigned char* zncc_output_rl = 0;
	unsigned char* crosscheck = 0;
	unsigned char* occlusionfill = 0;
	unsigned char* normalized = 0;
	unsigned width, height, resizedWidth, resizedHeight;
	int min = UCHAR_MAX;
	int max = 0;


	// read left input image from file
	clock_t startclk = clock();
	double start = queryProfiler();
	decodeImage(left, &image_l, &width, &height);
	resizedWidth = width / 4;
	resizedHeight = height / 4;
	clock_t endclk = clock();
	double end = queryProfiler();
	double elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to load the image_l: %lf seconds\n", elapsed_time);
	printf("real time used to load the image_l: %f seconds\n", end-start);


	// read right input image from file
	startclk = clock();
	start = queryProfiler();
	decodeImage(right, &image_r, &width, &height);
	endclk = clock();
	end = queryProfiler();
	elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to load the image_r: %lf seconds\n", elapsed_time);
	printf("real time used to load the image_r: %f  seconds\n", end-start);


	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id dev;
	cl_int err;
	cl_context context;

	// used for profiling
	cl_ulong kernelstarttimes[10];
	cl_ulong kernelendtimes[10];
	cl_double exectimems[10];
	cl_ulong bufferstarttimes[14];
	cl_ulong bufferendtimes[14];
	cl_double buffertimems[14];


	// get device info, implemented in file helpers.c
	if(printDeviceInfo(&dev, &platform) == EXIT_FAILURE)
	{
		return EXIT_FAILURE;
	}


	/* Create a context */
	context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	if(err < 0) {
		perror("Couldn't create a context");
		exit(1);   
	}


	/* Program data structures */
	cl_program program;
	FILE *program_handle;
	char *program_buffer[NUM_FILES];
	char *program_log;
	const char *file_name[] = {PROGRAM_FILE_0, PROGRAM_FILE_1, PROGRAM_FILE_2, PROGRAM_FILE_3, PROGRAM_FILE_4, PROGRAM_FILE_5, PROGRAM_FILE_6};
	const char options[] = "-cl-finite-math-only -cl-no-signed-zeros -cl-std=CL2.0";  
	size_t program_size[NUM_FILES];
	size_t log_size;

	/* Read each program file and place content into buffer array */
	for(unsigned i = 0; i < NUM_FILES; i++) {

		program_handle = fopen(file_name[i], "r");
		if(program_handle == NULL) {
			perror("Couldn't find the program file");
			return EXIT_FAILURE; 
		}
		fseek(program_handle, 0, SEEK_END);
		program_size[i] = ftell(program_handle);
		rewind(program_handle);
		program_buffer[i] = (char*)malloc(program_size[i]+1);
		program_buffer[i][program_size[i]] = '\0';
		fread(program_buffer[i], sizeof(char), program_size[i], 
				program_handle);
		fclose(program_handle);
	}

	/* Create a program containing all program content */
	program = clCreateProgramWithSource(context, NUM_FILES, 			
			(const char**)program_buffer, program_size, &err);
			
	if(err < 0) {
		perror("Couldn't create the program");
		return EXIT_FAILURE;   
	}

	/* Build program */
	err = clBuildProgram(program, 1, &dev, options, NULL, NULL);		
	if(err < 0) {

		/* Find size of log and print to std output */
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
				0, NULL, &log_size);
		program_log = (char*) malloc(log_size+1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
				log_size+1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		return EXIT_FAILURE;
	}

	// Allocate memory
	normalized =		(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);



	// Create command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err < 0) {
		perror("Couldn't create command queue");
		return EXIT_FAILURE;   
	}

	cl_event buffer_event_list[14];

	// Create memory buffers on the device
	cl_mem input_clmem_l = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem input_clmem_r = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("1 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem_l = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem_r = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("3 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem_l = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("4 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem_r = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("5 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem crosscheck_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("6 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem occlusionfill_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("7 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_output_clmem_lr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("8 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_output_clmem_rl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("9 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem normalized_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("10 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem min_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err < 0) {
		perror("11 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem max_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err < 0) {
		perror("12 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}

	// Copy buffers to the device
	err = clEnqueueWriteBuffer(command_queue, input_clmem_l, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image_l, 0, NULL, &buffer_event_list[0]);
	err = clEnqueueWriteBuffer(command_queue, input_clmem_r, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image_r, 0, NULL, &buffer_event_list[1]);
	err = clEnqueueWriteBuffer(command_queue, resized_clmem_l, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized_l, 0, NULL, &buffer_event_list[2]);
	err = clEnqueueWriteBuffer(command_queue, resized_clmem_r, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized_r, 0, NULL, &buffer_event_list[3]);
	err = clEnqueueWriteBuffer(command_queue, grayscale_clmem_l, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), grayscale_l, 0, NULL, &buffer_event_list[4]);
	err = clEnqueueWriteBuffer(command_queue, grayscale_clmem_r, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), grayscale_r, 0, NULL, &buffer_event_list[5]);
	err = clEnqueueWriteBuffer(command_queue, zncc_output_clmem_lr, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), zncc_output_lr, 0, NULL, &buffer_event_list[6]);
	err = clEnqueueWriteBuffer(command_queue, zncc_output_clmem_rl, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), zncc_output_rl, 0, NULL, &buffer_event_list[7]);
	err = clEnqueueWriteBuffer(command_queue, crosscheck_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), crosscheck, 0, NULL, &buffer_event_list[8]);
	err = clEnqueueWriteBuffer(command_queue, occlusionfill_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), occlusionfill, 0, NULL, &buffer_event_list[9]);
	err = clEnqueueWriteBuffer(command_queue, normalized_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), normalized, 0, NULL, &buffer_event_list[10]);
	err = clEnqueueWriteBuffer(command_queue, min_clmem, CL_TRUE, 0, sizeof(int), &min, 0, NULL, &buffer_event_list[11]);
	err = clEnqueueWriteBuffer(command_queue, max_clmem, CL_TRUE, 0, sizeof(int), &max, 0, NULL, &buffer_event_list[12]);
	
	if(err < 0) {
		perror("Couldn't copy memory buffers to the device");
		return EXIT_FAILURE;   
	}

	// Create kernels for operations
	cl_kernel resize_krnl_1 = clCreateKernel(program, KERNEL_NAME_0, &err);
	if(err < 0) {
		perror("0 Couldn't create resize kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel resize_krnl_2 = clCreateKernel(program, KERNEL_NAME_0, &err);
	if(err < 0) {
		perror("1 Couldn't create resize kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel grayscale_krnl_1 = clCreateKernel(program, KERNEL_NAME_1, &err);
	if(err < 0) {
		perror("2 Couldn't create grayscale kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel grayscale_krnl_2 = clCreateKernel(program, KERNEL_NAME_1, &err);
	if(err < 0) {
		perror("3 Couldn't create grayscale kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel zncc_krnl_1 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		perror("4 Couldn't create zncc kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel zncc_krnl_2 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		perror("5 Couldn't create zncc kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel crosscheck_krnl = clCreateKernel(program, KERNEL_NAME_3, &err);
	if(err < 0) {
		perror("6 Couldn't create crosscheck kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel occlusionfill_krnl = clCreateKernel(program, KERNEL_NAME_4, &err);
	if(err < 0) {
		perror("7 Couldn't create occlusion kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel findminmax_krnl = clCreateKernel(program, KERNEL_NAME_5, &err);
	if(err < 0) {
		perror("8 Couldn't create findminmax kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel normalize_krnl = clCreateKernel(program, KERNEL_NAME_6, &err);
	if(err < 0) {
		perror("9 Couldn't create normalize kernel");
		return EXIT_FAILURE;   
	}

	
	/* Set kernel arguments */
	
	// Resize kernels
	err = clSetKernelArg(resize_krnl_1, 0, sizeof(cl_mem), (void*)&input_clmem_l);
	if(err < 0) {
		perror("0 Couldn't set resize_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_1, 1, sizeof(cl_mem), (void*)&resized_clmem_l);
	if(err < 0) {
		perror("1 Couldn't set resize_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_2, 0, sizeof(cl_mem), (void*)&input_clmem_r);
	if(err < 0) {
		perror("2 Couldn't set resize_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_2, 1, sizeof(cl_mem), (void*)&resized_clmem_r);
	if(err < 0) {
		perror("3 Couldn't set resize_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// Grayscale kernels
	err = clSetKernelArg(grayscale_krnl_1, 0, sizeof(cl_mem), (void*)&resized_clmem_l);
	if(err < 0) {
		perror("4 Couldn't set grayscale_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_1, 1, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		perror("5 Couldn't set grayscale_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_2, 0, sizeof(cl_mem), (void*)&resized_clmem_r);
	if(err < 0) {
		perror("6 Couldn't set grayscale_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_2, 1, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		perror("7 Couldn't set grayscale_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// ZNCC kernels
	err = clSetKernelArg(zncc_krnl_1, 0, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		perror("8 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 1, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		perror("9 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 2, sizeof(cl_mem), (void*)&zncc_output_clmem_lr);
	if(err < 0) {
		perror("10 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		perror("11 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		perror("12 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		perror("13 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 6, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		perror("14 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 7, sizeof(unsigned int), &MAX_DISPARITY);
	if(err < 0) {
		perror("15 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 0, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		perror("16 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 1, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		perror("17 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 2, sizeof(cl_mem), (void*)&zncc_output_clmem_rl);
	if(err < 0) {
		perror("18 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		perror("19 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		perror("20 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		perror("21 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 6, sizeof(unsigned int), &MAX_DISPARITY_NEG);
	if(err < 0) {
		perror("22 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 7, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		perror("23 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// Crosscheck kernels
	err = clSetKernelArg(crosscheck_krnl, 0, sizeof(cl_mem), (void*)&zncc_output_clmem_lr);
	if(err < 0) {
		perror("24 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 1, sizeof(cl_mem), (void*)&zncc_output_clmem_rl);
	if(err < 0) {
		perror("25 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 2, sizeof(cl_mem), (void*)&crosscheck_clmem);
	if(err < 0) {
		perror("26 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 3, sizeof(int), &resizedWidth);
	if(err < 0) {
		perror("27 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 4, sizeof(int), &resizedHeight);
	if(err < 0) {
		perror("28 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 5, sizeof(int), &THRESHOLD);
	if(err < 0) {
		perror("29 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	// Oclusionfill kernels
	err = clSetKernelArg(occlusionfill_krnl, 0, sizeof(cl_mem), (void*)&crosscheck_clmem);
	if(err < 0) {
		perror("30 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 1, sizeof(cl_mem), (void*)&occlusionfill_clmem);
	if(err < 0) {
		perror("31 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 2, sizeof(int), &resizedWidth);
	if(err < 0) {
		perror("32 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 3, sizeof(int), &resizedHeight);
	if(err < 0) {
		perror("33 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 4, sizeof(int), &NEIGHBORHOOD_SIZE);
	if(err < 0) {
		perror("34 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(findminmax_krnl, 0, sizeof(cl_mem), &occlusionfill_clmem);
	if(err < 0) {
		perror("35 Couldn't set findminmax kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(findminmax_krnl, 1, sizeof(cl_mem), &min_clmem);
	if(err < 0) {
		perror("36 Couldn't set findminmax kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(findminmax_krnl, 2, sizeof(cl_mem), &max_clmem);
	if(err < 0) {
		perror("37 Couldn't set findminmax kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 0, sizeof(cl_mem), &occlusionfill_clmem);
	if(err < 0) {
		perror("38 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 1, sizeof(cl_mem), &normalized_clmem);
	if(err < 0) {
		perror("38 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 2, sizeof(cl_mem), &min_clmem);
	if(err < 0) {
		perror("39 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 3, sizeof(cl_mem), &max_clmem);
	if(err < 0) {
		perror("40 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	

	// Execute kernel on the device
	size_t global_size[2] = {(size_t)width*4, (size_t)height*4};
	size_t global_work_offset[2] = {0, 0};
	size_t global_size_resized[2] = {resizedWidth, resizedHeight};
   	size_t global_size_resized_asd = resizedWidth*resizedHeight*4;
	cl_event event_list[10];

	// Resize
	err = clEnqueueNDRangeKernel(command_queue, resize_krnl_1, 2, global_work_offset, global_size, NULL, 0, NULL, &event_list[0]);
	if(err < 0) {
		perror("0 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, resize_krnl_2, 2, global_work_offset, global_size, NULL, 1, event_list, &event_list[1]);
	if(err < 0) {
		perror("1 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// Grayscale
	err = clEnqueueNDRangeKernel(command_queue, grayscale_krnl_1, 1, NULL, &global_size_resized_asd, NULL, 2, event_list, &event_list[2]);
	if(err < 0) {
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, grayscale_krnl_2, 1, NULL, &global_size_resized_asd, NULL, 3, event_list, &event_list[3]);
	if(err < 0) {
		perror("3 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// ZNCC filter
	err = clEnqueueNDRangeKernel(command_queue, zncc_krnl_1, 2, NULL, global_size_resized, NULL, 4, event_list, &event_list[4]);
	if(err < 0) {
		perror("4 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, zncc_krnl_2, 2, NULL, global_size_resized, NULL, 4, event_list, &event_list[5]);
	if(err < 0) {
		perror("5 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// crosscheck
	err = clEnqueueNDRangeKernel(command_queue, crosscheck_krnl, 1, NULL, &global_size_resized_asd, NULL, 6, event_list, &event_list[6]);
	if(err < 0) {
		perror("6 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// occlusionfill
	err = clEnqueueNDRangeKernel(command_queue, occlusionfill_krnl, 2, NULL, global_size_resized, NULL, 7, event_list, &event_list[7]);
	if(err < 0) {
		perror("7 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// findminmax
	err = clEnqueueNDRangeKernel(command_queue, findminmax_krnl, 1, NULL, &global_size_resized_asd, NULL, 8, event_list, &event_list[8]);
	if(err < 0) {
		perror("8 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// normalize
	err = clEnqueueNDRangeKernel(command_queue, normalize_krnl, 1, NULL, &global_size_resized_asd, NULL, 9, event_list, &event_list[9]);
	if(err < 0) {
		perror("9 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	
	// Read results
	err = clEnqueueReadBuffer(command_queue, normalized_clmem, CL_TRUE, 0, resizedWidth*resizedHeight*sizeof(unsigned char), normalized, 10, event_list, &buffer_event_list[13]);
	if(err < 0) {
		perror("Error in clEnqueueReadBuffer");
		return EXIT_FAILURE;
	}

	// profile kernel execution times
	clWaitForEvents(10, (const cl_event*)&event_list);

	// resize left
	clGetEventProfilingInfo(event_list[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[0], NULL);
	clGetEventProfilingInfo(event_list[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[0], NULL);
	exectimems[0] = (cl_double)(kernelendtimes[0] - kernelstarttimes[0])*(cl_double)(1e-06);
	printf("resize kernel execution time for left image: %lf ms\n", exectimems[0]);

	// resize right
	clGetEventProfilingInfo(event_list[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[1], NULL);
	clGetEventProfilingInfo(event_list[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[1], NULL);
	exectimems[1] = (cl_double)(kernelendtimes[1] - kernelstarttimes[1])*(cl_double)(1e-06);
	printf("resize kernel execution time for right image: %lf ms\n", exectimems[1]);

	// grayscale left
	clGetEventProfilingInfo(event_list[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[2], NULL);
	clGetEventProfilingInfo(event_list[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[2], NULL);
	exectimems[2] = (cl_double)(kernelendtimes[2] - kernelstarttimes[2])*(cl_double)(1e-06);
	printf("grayscale kernel execution time for left image: %lf ms\n", exectimems[2]);

	// grayscale right
	clGetEventProfilingInfo(event_list[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[3], NULL);
	clGetEventProfilingInfo(event_list[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[3], NULL);
	exectimems[3] = (cl_double)(kernelendtimes[3] - kernelstarttimes[3])*(cl_double)(1e-06);
	printf("grayscale kernel execution time for right image: %lf ms\n", exectimems[3]);

	// zncc left
	clGetEventProfilingInfo(event_list[4], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[4], NULL);
	clGetEventProfilingInfo(event_list[4], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[4], NULL);
	exectimems[4] = (cl_double)(kernelendtimes[4] - kernelstarttimes[4])*(cl_double)(1e-06);
	printf("zncc kernel execution time for left image: %lf ms\n", exectimems[4]);

	// zncc right
	clGetEventProfilingInfo(event_list[5], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[5], NULL);
	clGetEventProfilingInfo(event_list[5], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[5], NULL);
	exectimems[5] = (cl_double)(kernelendtimes[5] - kernelstarttimes[5])*(cl_double)(1e-06);
	printf("zncc kernel execution time for right image: %lf ms\n", exectimems[5]);

	// crosscheck
	clGetEventProfilingInfo(event_list[6], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[6], NULL);
	clGetEventProfilingInfo(event_list[6], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[6], NULL);
	exectimems[6] = (cl_double)(kernelendtimes[6] - kernelstarttimes[6])*(cl_double)(1e-06);
	printf("crosscheck kernel execution time for right image: %lf ms\n", exectimems[6]);

	// occlusionfill
	clGetEventProfilingInfo(event_list[7], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[7], NULL);
	clGetEventProfilingInfo(event_list[7], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[7], NULL);
	exectimems[7] = (cl_double)(kernelendtimes[7] - kernelstarttimes[7])*(cl_double)(1e-06);
	printf("occlusionfill kernel execution time for right image: %lf ms\n", exectimems[7]);

	// findminmax
	clGetEventProfilingInfo(event_list[8], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[8], NULL);
	clGetEventProfilingInfo(event_list[8], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[8], NULL);
	exectimems[8] = (cl_double)(kernelendtimes[8] - kernelstarttimes[8])*(cl_double)(1e-06);
	printf("findminmax kernel execution time for right image: %lf ms\n", exectimems[8]);

	// normalize
	clGetEventProfilingInfo(event_list[9], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[9], NULL);
	clGetEventProfilingInfo(event_list[9], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[9], NULL);
	exectimems[9] = (cl_double)(kernelendtimes[9] - kernelstarttimes[9])*(cl_double)(1e-06);
	printf("normalize kernel execution time for right image: %lf ms\n", exectimems[9]);


	// buffer writes
	clGetEventProfilingInfo(buffer_event_list[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &bufferstarttimes[0], NULL);
	clGetEventProfilingInfo(buffer_event_list[12], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &bufferendtimes[12], NULL);
	buffertimems[0] = (cl_double)(bufferendtimes[12] - bufferstarttimes[0])*(cl_double)(1e-06);
	printf("time taken for writing buffer to device: %lf ms\n", buffertimems[0]);

	// buffer reads
	clGetEventProfilingInfo(buffer_event_list[13], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &bufferstarttimes[13], NULL);
	clGetEventProfilingInfo(buffer_event_list[13], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &bufferendtimes[13], NULL);
	buffertimems[13] = (cl_double)(bufferendtimes[13] - bufferstarttimes[13])*(cl_double)(1e-06);
	printf("time taken for reading buffers from device: %lf ms\n", buffertimems[13]);

	// Clean up and wait for all the commands to complete
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	if(err < 0) {
		perror("Error executing command queue");
		return EXIT_FAILURE;   
	}
	
	// output result
	startclk = clock();
	start = queryProfiler();
   	lodepng_encode_file(outputimg, normalized, resizedWidth, resizedHeight, LCT_GREY, 8);
	endclk = clock();
	end = queryProfiler();
	elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to encode image: %lf seconds\n", elapsed_time);
	printf("real time used to encode image: %f  seconds\n", end-start);


	/* Deallocate resources */
	for(unsigned i = 0; i < NUM_FILES; i++) {
		free(program_buffer[i]);
	}
	err = clReleaseKernel(grayscale_krnl_1);
	err = clReleaseKernel(grayscale_krnl_2);
	err = clReleaseKernel(zncc_krnl_1);
	err = clReleaseKernel(resize_krnl_1);
	err = clReleaseKernel(resize_krnl_2);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(input_clmem_l);
	err = clReleaseMemObject(input_clmem_r);
	err = clReleaseMemObject(grayscale_clmem_l);
	err = clReleaseMemObject(grayscale_clmem_r);
	err = clReleaseMemObject(resized_clmem_l);
	err = clReleaseMemObject(resized_clmem_r);
	err = clReleaseMemObject(crosscheck_clmem);
	err = clReleaseMemObject(occlusionfill_clmem);
	err = clReleaseMemObject(zncc_output_clmem_lr);
	err = clReleaseMemObject(zncc_output_clmem_rl);
	err = clReleaseMemObject(normalized_clmem);
	err = clReleaseMemObject(min_clmem);
	err = clReleaseMemObject(max_clmem);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	if(err < 0) {
		perror("Error deallocating resources");
		return EXIT_FAILURE;   
	}
	
	free(normalized);


	printf("\n\nProgram finished\n");

	clock_t endprogclk = clock();
	double endprog = queryProfiler();

	double elapsed_time_prog = (endprogclk-startprogclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken by program execution: %lf seconds\n", elapsed_time_prog);
	printf("real time taken by program execution: %f  seconds\n", endprog-startprog);

	return EXIT_SUCCESS;
}


