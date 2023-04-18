
#define NUM_FILES 5
#define PROGRAM_FILE_0 "kernels/zncc.cl"
#define PROGRAM_FILE_1 "kernels/grayscale.cl"
#define PROGRAM_FILE_2 "kernels/resizeimage.cl"
#define PROGRAM_FILE_3 "kernels/crosscheck.cl"
#define PROGRAM_FILE_4 "kernels/occlusionfill.cl"
#define KERNEL_NAME_0 "resizeimage"
#define KERNEL_NAME_1 "grayscale"
#define KERNEL_NAME_2 "zncc"
#define KERNEL_NAME_3 "crosscheck"
#define KERNEL_NAME_4 "occlusionfill"


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

// ZNCC PARAMETERS
int MIN_DISPARITY = 0;
int MAX_DISPARITY = 65;
int MAX_DISPARITY_NEG = -65;
int B = 10;
// CROSSCHECK PARAMETERS
int THRESHOLD = 2;
// OCCLUSIONFILL PARAMETERS
int NEIGHBORHOOD_SIZE = 256;

void normalize(unsigned char* disparity_img, unsigned width, unsigned height) {

	unsigned max = 0;
	unsigned min = UCHAR_MAX;

	for (unsigned i = 0; i < width*height; i++) {
		if (disparity_img[i] > max) {
			max = disparity_img[i];
		}
		if (disparity_img[i] < min) {
			min = disparity_img[i];
		}
	}
	for (unsigned i = 0; i < width*height; i++) {
		disparity_img[i] = 255*(disparity_img[i]- min)/(max-min);
	}
}


int main(int argc, char **argv)
	{
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
	unsigned width, height, resizedWidth, resizedHeight;


	clock_t start = clock();
	decodeImage(left, &image_l, &width, &height);
	resizedWidth = width / 4;
	resizedHeight = height / 4;
	clock_t end = clock();
	double elapsed_time = (end-start)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to load the image_l: %lf seconds", elapsed_time);


	start = clock();
	decodeImage(right, &image_r, &width, &height);
	end = clock();
	elapsed_time = (end-start)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to load the image_r: %lf seconds", elapsed_time);


	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id dev;
	cl_int err;
	cl_context context;

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
	const char *file_name[] = {PROGRAM_FILE_0, PROGRAM_FILE_1, PROGRAM_FILE_2, PROGRAM_FILE_3, PROGRAM_FILE_4};
	const char options[] = "-cl-finite-math-only -cl-no-signed-zeros";  
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
	resized_l = 		(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	resized_r = 		(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	grayscale_l = 		(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	grayscale_r = 		(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	crosscheck = 		(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	occlusionfill = 	(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	zncc_output_lr = 	(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);
	zncc_output_rl = 	(unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);

	// Create command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, dev, 0, &err);
	if(err < 0) {
		perror("Couldn't create command queue");
		return EXIT_FAILURE;   
	}

	// Create memory buffers on the device
	cl_mem input_clmem_l = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem input_clmem_r = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem_l = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem_r = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem_l = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("1 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem_r = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("1 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem crosscheck_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem occlusionfill_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_output_clmem_lr = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem zncc_output_clmem_rl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}

	// Copy buffers to the device
	err = clEnqueueWriteBuffer(command_queue, input_clmem_l, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image_l, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, input_clmem_r, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, resized_clmem_l, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized_l, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, resized_clmem_r, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, grayscale_clmem_l, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), grayscale_l, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, grayscale_clmem_r, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), grayscale_r, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, zncc_output_clmem_lr, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), zncc_output_lr, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, zncc_output_clmem_rl, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), zncc_output_rl, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, crosscheck_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), crosscheck, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, occlusionfill_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), occlusionfill, 0, NULL, NULL);
	
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
		perror("0 Couldn't create grayscale kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel grayscale_krnl_2 = clCreateKernel(program, KERNEL_NAME_1, &err);
	if(err < 0) {
		perror("1 Couldn't create grayscale kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel zncc_krnl_1 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		perror("0 Couldn't create zncc kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel zncc_krnl_2 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		perror("1 Couldn't create zncc kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel crosscheck_krnl = clCreateKernel(program, KERNEL_NAME_3, &err);
	if(err < 0) {
		perror("0 Couldn't create crosscheck kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel occlusionfill_krnl = clCreateKernel(program, KERNEL_NAME_4, &err);
	if(err < 0) {
		perror("0 Couldn't create occlusion kernel");
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
		perror("0 Couldn't set resize_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(resize_krnl_2, 1, sizeof(cl_mem), (void*)&resized_clmem_r);
	if(err < 0) {
		perror("1 Couldn't set resize_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// Grayscale kernels
	err = clSetKernelArg(grayscale_krnl_1, 0, sizeof(cl_mem), (void*)&resized_clmem_l);
	if(err < 0) {
		perror("0 Couldn't set grayscale_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_1, 1, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		perror("1 Couldn't set grayscale_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_2, 0, sizeof(cl_mem), (void*)&resized_clmem_r);
	if(err < 0) {
		perror("0 Couldn't set grayscale_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(grayscale_krnl_2, 1, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		perror("1 Couldn't set grayscale_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// ZNCC kernels
	err = clSetKernelArg(zncc_krnl_1, 0, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		perror("0 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 1, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		perror("1 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 2, sizeof(cl_mem), (void*)&zncc_output_clmem_lr);
	if(err < 0) {
		perror("2 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		perror("3 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		perror("4 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		perror("5 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 6, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		perror("6 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_1, 7, sizeof(unsigned int), &MAX_DISPARITY);
	if(err < 0) {
		perror("7 Couldn't set zncc_krnl_1 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 0, sizeof(cl_mem), (void*)&grayscale_clmem_r);
	if(err < 0) {
		perror("0 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 1, sizeof(cl_mem), (void*)&grayscale_clmem_l);
	if(err < 0) {
		perror("1 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 2, sizeof(cl_mem), (void*)&zncc_output_clmem_rl);
	if(err < 0) {
		perror("2 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		perror("3 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		perror("4 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		perror("5 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 6, sizeof(unsigned int), &MAX_DISPARITY_NEG);
	if(err < 0) {
		perror("6 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(zncc_krnl_2, 7, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		perror("7 Couldn't set zncc_krnl_2 arguments");
		return EXIT_FAILURE;   
	}
	// Crosscheck kernels
	err = clSetKernelArg(crosscheck_krnl, 0, sizeof(cl_mem), (void*)&zncc_output_clmem_lr);
	if(err < 0) {
		perror("0 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 1, sizeof(cl_mem), (void*)&zncc_output_clmem_rl);
	if(err < 0) {
		perror("1 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 2, sizeof(cl_mem), (void*)&crosscheck_clmem);
	if(err < 0) {
		perror("2 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 3, sizeof(int), &resizedWidth);
	if(err < 0) {
		perror("3 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 4, sizeof(int), &resizedHeight);
	if(err < 0) {
		perror("4 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(crosscheck_krnl, 5, sizeof(int), &THRESHOLD);
	if(err < 0) {
		perror("5 Couldn't set crosscheck kernel arguments");
		return EXIT_FAILURE;   
	}
	// Oclusionfill kernels
	err = clSetKernelArg(occlusionfill_krnl, 0, sizeof(cl_mem), (void*)&crosscheck_clmem);
	if(err < 0) {
		perror("0 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 1, sizeof(cl_mem), (void*)&occlusionfill_clmem);
	if(err < 0) {
		perror("1 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 2, sizeof(int), &resizedWidth);
	if(err < 0) {
		perror("2 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 3, sizeof(int), &resizedHeight);
	if(err < 0) {
		perror("3 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(occlusionfill_krnl, 4, sizeof(int), &NEIGHBORHOOD_SIZE);
	if(err < 0) {
		perror("4 Couldn't set occlusionfill kernel arguments");
		return EXIT_FAILURE;   
	}
	


	// Execute kernel on the device
	size_t global_size[2] = {(size_t)width*4, (size_t)height*4};
	size_t global_work_offset[2] = {0, 0};
	size_t global_size_resized[2] = {resizedWidth, resizedHeight};
   	size_t global_size_resized_asd = resizedWidth*resizedHeight*4;
	cl_event event_list[11];

	// Resize
	err = clEnqueueNDRangeKernel(command_queue, resize_krnl_1, 2, global_work_offset, global_size, NULL, 0, NULL, &event_list[0]);
	if(err < 0) {
		perror("0 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, resize_krnl_2, 2, global_work_offset, global_size, NULL, 1, event_list, &event_list[1]);
	if(err < 0) {
		perror("0 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// Grayscale
	err = clEnqueueNDRangeKernel(command_queue, grayscale_krnl_1, 1, NULL, &global_size_resized_asd, NULL, 2, event_list, &event_list[2]);
	if(err < 0) {
		perror("1 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, grayscale_krnl_2, 1, NULL, &global_size_resized_asd, NULL, 3, event_list, &event_list[3]);
	if(err < 0) {
		perror("1 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// ZNCC filter
	err = clEnqueueNDRangeKernel(command_queue, zncc_krnl_1, 2, NULL, global_size_resized, NULL, 4, event_list, &event_list[4]);
	if(err < 0) {
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, zncc_krnl_2, 2, NULL, global_size_resized, NULL, 4, event_list, &event_list[5]);
	if(err < 0) {
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// crosscheck
	err = clEnqueueNDRangeKernel(command_queue, crosscheck_krnl, 1, NULL, &global_size_resized_asd, NULL, 6, event_list, &event_list[6]);
	if(err < 0) {
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// occlusionfill
	err = clEnqueueNDRangeKernel(command_queue, occlusionfill_krnl, 2, NULL, global_size_resized, NULL, 7, event_list, &event_list[7]);
	if(err < 0) {
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	
	// Read results
	err = clEnqueueReadBuffer(command_queue, occlusionfill_clmem, CL_TRUE, 0, resizedWidth*resizedHeight*sizeof(unsigned char), occlusionfill, 8, event_list, NULL);
	if(err < 0) {
		perror("Error in clEnqueueReadBuffer");
		return EXIT_FAILURE;
	}



	// Clean up and wait for all the commands to complete
	err = clFlush(command_queue);
	err = clFinish(command_queue);
	if(err < 0) {
		perror("Error executing command queue");
		return EXIT_FAILURE;   
	}
	
	// output result
	start = clock();
	normalize(occlusionfill, resizedWidth, resizedHeight);
   	lodepng_encode_file(outputimg, occlusionfill, resizedWidth, resizedHeight, LCT_GREY, 8);
	end = clock();
	elapsed_time = (end-start)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to encode image: %lf seconds", elapsed_time);


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
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	if(err < 0) {
		perror("Error deallocating resources");
		return EXIT_FAILURE;   
	}
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


	printf("\n\nProgram finished\n");
	return EXIT_SUCCESS;
	}


