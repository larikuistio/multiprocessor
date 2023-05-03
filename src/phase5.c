
#define NUM_FILES 7
#define PROGRAM_FILE_0 "kernels/zncc_new.cl"
#define PROGRAM_FILE_1 "kernels/grayscale.cl"
#define PROGRAM_FILE_2 "kernels/resizeimage.cl"
#define PROGRAM_FILE_3 "kernels/crosscheck.cl"
#define PROGRAM_FILE_4 "kernels/occlusionfill.cl"
#define PROGRAM_FILE_5 "kernels/findminmax.cl"
#define PROGRAM_FILE_6 "kernels/normalize.cl"
#define KERNEL_NAME_0 "resizeimage"
#define KERNEL_NAME_1 "grayscale"
#define KERNEL_NAME_2 "zncc"
#define KERNEL_NAME_3 "crosscheck"
#define KERNEL_NAME_4 "occlusionfill"
#define KERNEL_NAME_5 "findminmax"
#define KERNEL_NAME_6 "normalize"


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
#include <stdatomic.h>


// ZNCC PARAMETERS
int MIN_DISPARITY = 0;
int MAX_DISPARITY = 65;
int MAX_DISPARITY_NEG = -65;
int B = 8;
// CROSSCHECK PARAMETERS
int THRESHOLD = 2;
// OCCLUSIONFILL PARAMETERS
int NEIGHBORHOOD_SIZE = 8;




int main(int argc, char **argv)
{

	clock_t startprogclk = clock();
	double startprog = queryProfiler();

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
	int* disparities_lr = 0;
	int* disparities_rl = 0;
	double* zncc_scores_lr = 0;
	double* zncc_scores_rl = 0;


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

	cl_ulong kernelstarttimes[10];
	cl_ulong kernelendtimes[10];
	cl_double exectimems[10];

	cl_ulong znccstarttimes[2][28];
	cl_ulong znccendtimes[2][28];
	cl_double znccexectimesms[2][28];

	cl_ulong bufferstarttimes[18];
	cl_ulong bufferendtimes[18];
	cl_double buffertimems[18];

	if(printDeviceInfo(&dev, &platform) == EXIT_FAILURE)
	{
		return EXIT_FAILURE;
	}


	/* Create a context */
	context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("Couldn't create a context");
		exit(1);   
	}


	/* Program data structures */
	cl_program program;
	FILE *program_handle;
	char *program_buffer[NUM_FILES];
	char *program_log;
	const char *file_name[] = {PROGRAM_FILE_0, PROGRAM_FILE_1, PROGRAM_FILE_2, PROGRAM_FILE_3, PROGRAM_FILE_4, PROGRAM_FILE_5, PROGRAM_FILE_6};
	const char options[] = "-cl-finite-math-only -cl-no-signed-zeros -cl-std=CL2.0 -g -Werror -cl-opt-disable";  
	size_t program_size[NUM_FILES];
	size_t log_size;

	/* Read each program file and place content into buffer array */
	for(unsigned i = 0; i < NUM_FILES; i++) {

		program_handle = fopen(file_name[i], "r");
		if(program_handle == NULL) {
			printOpenCLErrorCode(err);
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
		printOpenCLErrorCode(err);
		perror("Couldn't create the program");
		return EXIT_FAILURE;   
	}

	/* Build program */
	err = clBuildProgram(program, 1, &dev, options, NULL, NULL);		
	if(err < 0) {
		printOpenCLErrorCode(err);
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
	image_l = (unsigned char*)malloc(sizeof(unsigned char)*width*height);
	image_r = (unsigned char*)malloc(sizeof(unsigned char)*width*height);
	resized_l = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	resized_r = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	grayscale_l = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	grayscale_r = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	crosscheck = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	occlusionfill = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	zncc_output_lr = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	zncc_output_rl = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	disparities_lr = (int*)malloc(sizeof(int)*resizedWidth*resizedHeight);
	disparities_rl = (int*)malloc(sizeof(int)*resizedWidth*resizedHeight);
	zncc_scores_lr = (double*)malloc(sizeof(double)*resizedWidth*resizedHeight);
	zncc_scores_rl = (double*)malloc(sizeof(double)*resizedWidth*resizedHeight);


	// Create command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("Couldn't create command queue");
		return EXIT_FAILURE;   
	}

	cl_event buffer_event_list[18];

	// Create memory buffers on the device
	cl_mem input_clmem_l = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem input_clmem_r = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("1 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem_l = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem_r = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("3 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem_l = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("4 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem_r = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("5 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem crosscheck_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("6 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem occlusionfill_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("7 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_output_clmem_lr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("8 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_output_clmem_rl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("9 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem normalized_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("10 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem min_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("11 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem max_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("12 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem disparities_clmem_lr = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(int), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("13 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_scores_clmem_lr = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(double), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("14 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem disparities_clmem_rl = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(int), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("15 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_scores_clmem_rl = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(double), NULL, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("16 Couldn't create memory buffers on the device");
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
	err = clEnqueueWriteBuffer(command_queue, disparities_clmem_lr, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(int), disparities_lr, 0, NULL, &buffer_event_list[13]);
	err = clEnqueueWriteBuffer(command_queue, zncc_scores_clmem_lr, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(double), zncc_scores_lr, 0, NULL, &buffer_event_list[14]);
	err = clEnqueueWriteBuffer(command_queue, disparities_clmem_rl, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(int), disparities_rl, 0, NULL, &buffer_event_list[15]);
	err = clEnqueueWriteBuffer(command_queue, zncc_scores_clmem_rl, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(double), zncc_scores_rl, 0, NULL, &buffer_event_list[16]);
	
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("Couldn't copy memory buffers to the device");
		return EXIT_FAILURE;   
	}

	// Create kernels for operations
	cl_kernel resize_krnl_1 = clCreateKernel(program, KERNEL_NAME_0, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("0 Couldn't create resize kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel resize_krnl_2 = clCreateKernel(program, KERNEL_NAME_0, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("1 Couldn't create resize kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel grayscale_krnl_1 = clCreateKernel(program, KERNEL_NAME_1, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("2 Couldn't create grayscale kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel grayscale_krnl_2 = clCreateKernel(program, KERNEL_NAME_1, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("3 Couldn't create grayscale kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel zncc_krnl_1 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("4 Couldn't create zncc kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel zncc_krnl_2 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("5 Couldn't create zncc kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel crosscheck_krnl = clCreateKernel(program, KERNEL_NAME_3, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("6 Couldn't create crosscheck kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel occlusionfill_krnl = clCreateKernel(program, KERNEL_NAME_4, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("7 Couldn't create occlusion kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel findminmax_krnl = clCreateKernel(program, KERNEL_NAME_5, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("8 Couldn't create findminmax kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel normalize_krnl = clCreateKernel(program, KERNEL_NAME_6, &err);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("9 Couldn't create normalize kernel");
		return EXIT_FAILURE;   
	}

	
	/* Set kernel arguments */
	
	// Resize kernels
	err = clSetKernelArg(resize_krnl_1, 0, sizeof(cl_mem), (void*)&input_clmem_l);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("0 Couldn't set resize_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_1, 1, sizeof(cl_mem), (void*)&resized_clmem_l);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("1 Couldn't set resize_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_2, 0, sizeof(cl_mem), (void*)&input_clmem_r);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("2 Couldn't set resize_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_2, 1, sizeof(cl_mem), (void*)&resized_clmem_r);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("3 Couldn't set resize_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// Grayscale kernels
	err = clSetKernelArg(grayscale_krnl_1, 0, sizeof(cl_mem), (void*)&resized_clmem_l);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("4 Couldn't set grayscale_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_1, 1, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("5 Couldn't set grayscale_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_2, 0, sizeof(cl_mem), (void*)&resized_clmem_r);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("6 Couldn't set grayscale_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_2, 1, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("7 Couldn't set grayscale_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// ZNCC kernels
	err = clSetKernelArg(zncc_krnl_1, 0, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("8 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 1, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("9 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 2, sizeof(cl_mem), (void*)&zncc_output_clmem_lr);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("10 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("11 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("12 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("13 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 6, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("14 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 7, sizeof(unsigned int), &MAX_DISPARITY);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("15 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 8, sizeof(cl_mem), (void*)&disparities_clmem_lr);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("110 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 9, sizeof(cl_mem), (void*)&zncc_scores_clmem_lr);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("111 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 10, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("112 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 11, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("113 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 12, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("114 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 13, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("115 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 14, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("116 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 0, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("16 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 1, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("17 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 2, sizeof(cl_mem), (void*)&zncc_output_clmem_rl);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("18 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("19 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("20 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("21 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 6, sizeof(unsigned int), &MAX_DISPARITY_NEG);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("22 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 7, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("23 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 8, sizeof(cl_mem), (void*)&disparities_clmem_rl);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("120 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 9, sizeof(cl_mem), (void*)&zncc_scores_clmem_rl);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("121 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 10, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("122 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 11, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("123 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 12, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("124 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 13, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("125 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 14, sizeof(double), NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("126 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	
	// Crosscheck kernels
	err = clSetKernelArg(crosscheck_krnl, 0, sizeof(cl_mem), (void*)&zncc_output_clmem_lr);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("24 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 1, sizeof(cl_mem), (void*)&zncc_output_clmem_rl);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("25 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 2, sizeof(cl_mem), (void*)&crosscheck_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("26 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 3, sizeof(int), &resizedWidth);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("27 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 4, sizeof(int), &resizedHeight);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("28 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 5, sizeof(int), &THRESHOLD);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("29 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	// Oclusionfill kernels
	err = clSetKernelArg(occlusionfill_krnl, 0, sizeof(cl_mem), (void*)&crosscheck_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("30 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 1, sizeof(cl_mem), (void*)&occlusionfill_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("31 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 2, sizeof(int), &resizedWidth);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("32 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 3, sizeof(int), &resizedHeight);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("33 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 4, sizeof(int), &NEIGHBORHOOD_SIZE);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("34 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(findminmax_krnl, 0, sizeof(cl_mem), &occlusionfill_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("35 Couldn't set findminmax kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(findminmax_krnl, 1, sizeof(cl_mem), &min_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("36 Couldn't set findminmax kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(findminmax_krnl, 2, sizeof(cl_mem), &max_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("37 Couldn't set findminmax kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 0, sizeof(cl_mem), &occlusionfill_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("38 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 1, sizeof(cl_mem), &normalized_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("38 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 2, sizeof(cl_mem), &min_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("39 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(normalize_krnl, 3, sizeof(cl_mem), &max_clmem);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("40 Couldn't set normalize kernel arguments");
		return EXIT_FAILURE;   
	}
	
	size_t kernel_work_group_size;
	err = clGetKernelWorkGroupInfo(zncc_krnl_1, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_work_group_size, NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("0 Error checking kernel work group size");
		return EXIT_FAILURE;   
	}
	printf("zncc_krnl_1 work group size: %ld\n", kernel_work_group_size);
	err = clGetKernelWorkGroupInfo(zncc_krnl_2, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kernel_work_group_size, NULL);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("1 Error checking kernel work group size");
		return EXIT_FAILURE;   
	}
	printf("zncc_krnl_2 work group size: %ld\n", kernel_work_group_size);
	 
	// Execute kernel on the device
	const size_t global_size[2] = {(size_t)width*4, (size_t)height*4};
	const size_t global_work_offset[2] = {0, 0};
	const size_t global_size_resized[2] = {(size_t)resizedWidth, (size_t)resizedHeight};
	const size_t global_size_zncc_lr[3] = {(size_t)resizedWidth/7, (size_t)resizedHeight/4, (size_t)MAX_DISPARITY};
	const size_t global_size_zncc_rl[3] = {(size_t)resizedWidth/7, (size_t)resizedHeight/4, (size_t)MAX_DISPARITY};
	size_t global_work_offset_zncc[7][4][3];
	for(size_t i = 0; i < 7; i++)
	{
		for(size_t j = 0; j < 4; j++)
		{
			global_work_offset_zncc[i][j][0] = i * (resizedWidth/7);
			global_work_offset_zncc[i][j][1] = j * (resizedHeight/4);
			global_work_offset_zncc[i][j][2] = 0;
		}
	}

	const size_t local_size[3] = {(size_t)B, (size_t)B, (size_t)1};
   	const size_t global_size_resized_asd = (size_t)resizedWidth*resizedHeight*4;
	cl_event event_list[10];
	cl_event zncc_event_list[2][28];
	 
	// Resize
	err = clEnqueueNDRangeKernel(command_queue, resize_krnl_1, 2, NULL, global_size, NULL, 0, NULL, &event_list[0]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("0 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, resize_krnl_2, 2, NULL, global_size, NULL, 1, event_list, &event_list[1]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("1 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// Grayscale
	err = clEnqueueNDRangeKernel(command_queue, grayscale_krnl_1, 1, NULL, &global_size_resized_asd, NULL, 2, event_list, &event_list[2]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, grayscale_krnl_2, 1, NULL, &global_size_resized_asd, NULL, 3, event_list, &event_list[3]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("3 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
 
	size_t jj = 0;
	// ZNCC filter
	for(size_t i = 0; i < 7; i++)
	{
		for(size_t j = 0; j < 4; j++)
		{
			err = clEnqueueNDRangeKernel(command_queue, zncc_krnl_1, 3, global_work_offset_zncc[i][j], global_size_zncc_lr, local_size, 4, event_list, &zncc_event_list[0][i*4 + j]);
			if(err < 0) {
				printOpenCLErrorCode(err);
				perror("4 Error in clEnqueueNDRangeKernel");
				return EXIT_FAILURE;   
			}
			err = clEnqueueNDRangeKernel(command_queue, zncc_krnl_2, 3, global_work_offset_zncc[i][j], global_size_zncc_rl, local_size, 4, event_list, &zncc_event_list[1][i*4 + j]);
			if(err < 0) {
				printOpenCLErrorCode(err);
				perror("5 Error in clEnqueueNDRangeKernel");
				return EXIT_FAILURE;   
			}
			jj = j;
		}
		clWaitForEvents(i*4+jj, (const cl_event*)&zncc_event_list[0]);
		clWaitForEvents(i*4+jj, (const cl_event*)&zncc_event_list[1]);
	}
	clWaitForEvents(28, (const cl_event*)&zncc_event_list[0]);
	clWaitForEvents(28, (const cl_event*)&zncc_event_list[1]);
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("1 Error in clFinish");
		return EXIT_FAILURE;   
	}
 
	// crosscheck
	err = clEnqueueNDRangeKernel(command_queue, crosscheck_krnl, 1, NULL, &global_size_resized_asd, NULL, 4, event_list, &event_list[4]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("6 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// occlusionfill
	err = clEnqueueNDRangeKernel(command_queue, occlusionfill_krnl, 2, NULL, global_size_resized, NULL, 5, event_list, &event_list[5]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("7 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// findminmax
	err = clEnqueueNDRangeKernel(command_queue, findminmax_krnl, 1, NULL, &global_size_resized_asd, NULL, 6, event_list, &event_list[6]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("8 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// normalize
	err = clEnqueueNDRangeKernel(command_queue, normalize_krnl, 1, NULL, &global_size_resized_asd, NULL, 7, event_list, &event_list[7]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("9 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	
	// Read results

	clWaitForEvents(17, (const cl_event*)&buffer_event_list);
	clWaitForEvents(8, (const cl_event*)&event_list);
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("2 Error in clFinish");
		return EXIT_FAILURE;   
	}
	err = clEnqueueReadBuffer(command_queue, normalized_clmem, CL_TRUE, 0, resizedWidth*resizedHeight*sizeof(unsigned char), normalized, 0, NULL, &buffer_event_list[17]);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("Error in clEnqueueReadBuffer");
		return EXIT_FAILURE;
	}
	
	// profile kernel execution times
	
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
	clGetEventProfilingInfo(zncc_event_list[0][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &znccstarttimes[0][0], NULL);
	clGetEventProfilingInfo(zncc_event_list[0][27], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &znccendtimes[0][27], NULL);
	znccexectimesms[0][0] = (cl_double)(znccendtimes[0][27] - znccstarttimes[0][0])*(cl_double)(1e-06);
	printf("zncc kernels execution time for left image: %lf ms\n", znccexectimesms[0][0]);

	// zncc right
	clGetEventProfilingInfo(zncc_event_list[1][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &znccstarttimes[1][0], NULL);
	clGetEventProfilingInfo(zncc_event_list[1][27], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &znccendtimes[1][27], NULL);
	znccexectimesms[1][0] = (cl_double)(znccendtimes[1][27] - znccstarttimes[1][0])*(cl_double)(1e-06);
	printf("zncc kernels execution time for right image: %lf ms\n", znccexectimesms[1][0]);

	// crosscheck
	clGetEventProfilingInfo(event_list[4], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[4], NULL);
	clGetEventProfilingInfo(event_list[4], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[4], NULL);
	exectimems[4] = (cl_double)(kernelendtimes[4] - kernelstarttimes[4])*(cl_double)(1e-06);
	printf("crosscheck kernel execution time for right image: %lf ms\n", exectimems[4]);

	// occlusionfill
	clGetEventProfilingInfo(event_list[5], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[5], NULL);
	clGetEventProfilingInfo(event_list[5], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[5], NULL);
	exectimems[5] = (cl_double)(kernelendtimes[5] - kernelstarttimes[5])*(cl_double)(1e-06);
	printf("occlusionfill kernel execution time for right image: %lf ms\n", exectimems[5]);

	// findminmax
	clGetEventProfilingInfo(event_list[6], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[6], NULL);
	clGetEventProfilingInfo(event_list[6], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[6], NULL);
	exectimems[6] = (cl_double)(kernelendtimes[6] - kernelstarttimes[6])*(cl_double)(1e-06);
	printf("findminmax kernel execution time for right image: %lf ms\n", exectimems[6]);

	// normalize
	clGetEventProfilingInfo(event_list[7], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[7], NULL);
	clGetEventProfilingInfo(event_list[7], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[7], NULL);
	exectimems[7] = (cl_double)(kernelendtimes[7] - kernelstarttimes[7])*(cl_double)(1e-06);
	printf("normalize kernel execution time for right image: %lf ms\n", exectimems[7]);

	
	// profile buffers
	clWaitForEvents(18, (const cl_event*)&buffer_event_list);

	// buffer writes
	clGetEventProfilingInfo(buffer_event_list[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &bufferstarttimes[0], NULL);
	clGetEventProfilingInfo(buffer_event_list[16], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &bufferendtimes[16], NULL);
	buffertimems[0] = (cl_double)(bufferendtimes[16] - bufferstarttimes[0])*(cl_double)(1e-06);
	printf("time taken for writing buffer to device: %lf ms\n", buffertimems[0]);

	// buffer reads
	clGetEventProfilingInfo(buffer_event_list[17], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &bufferstarttimes[17], NULL);
	clGetEventProfilingInfo(buffer_event_list[17], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &bufferendtimes[17], NULL);
	buffertimems[17] = (cl_double)(bufferendtimes[17] - bufferstarttimes[17])*(cl_double)(1e-06);
	printf("time taken for reading buffers from device: %lf ms\n", buffertimems[17]);

	// Clean up and wait for all the commands to complete
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	if(err < 0) {
		printOpenCLErrorCode(err);
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
	err = clReleaseMemObject(disparities_clmem_lr);
	err = clReleaseMemObject(zncc_scores_clmem_lr);
	err = clReleaseMemObject(disparities_clmem_rl);
	err = clReleaseMemObject(zncc_scores_clmem_rl);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	if(err < 0) {
		printOpenCLErrorCode(err);
		perror("Error deallocating resources");
		return EXIT_FAILURE;   
	}
	
	free(normalized);
	free(image_l);
	free(image_r);
	free(resized_l);
	free(resized_r);
	free(grayscale_l);
	free(grayscale_r);
	free(crosscheck);
	free(occlusionfill);
	free(zncc_output_lr);
	free(zncc_output_rl);
	free(disparities_lr);
	free(disparities_rl);
	free(zncc_scores_lr);
	free(zncc_scores_rl);


	printf("\n\nProgram finished\n");

	clock_t endprogclk = clock();
	double endprog = queryProfiler();

	double elapsed_time_prog = (endprogclk-startprogclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken by program execution: %lf seconds\n", elapsed_time_prog);
	printf("real time taken by program execution: %f  seconds\n", endprog-startprog);

	return EXIT_SUCCESS;
}


