#define NUM_FILES 3
#define PROGRAM_FILE_0 "kernels/zncc.cl"
#define PROGRAM_FILE_1 "kernels/grayscale.cl"
#define PROGRAM_FILE_2 "kernels/resizeimage.cl"
#define KERNEL_NAME_0 "grayscale"
#define KERNEL_NAME_1 "zncc"
#define KERNEL_NAME_2 "resizeimage"


#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include "helpers.h"


int main(int argc, char **argv)
	{
	if (argc < 2) {
		printf("provide input and output image files names as arguments\n");
		return EXIT_FAILURE;
	}

	const char* left = argv[1];
	const char* right = argv[2];
	const char* outputimg = argv[3];
	unsigned char* image1 = 0;
	unsigned char* image2 = 0;
	unsigned char* grayscale1 = 0;
	unsigned char* grayscale2 = 0;
	unsigned char* output = 0;
	unsigned char* resized1 = 0;
	unsigned char* resized2 = 0;
	unsigned width, height, resizedWidth, resizedHeight;

	int MAX_DISPARITY = 65;
	int MIN_DISPARITY = 0;
	int B = 5;

	clock_t start = clock();
	decodeImage(left, &image1, &width, &height);
	resizedWidth = width / 4;
	resizedHeight = height / 4;
	clock_t end = clock();
	double elapsed_time = (end-start)/(double)CLOCKS_PER_SEC;
	printf("cpu time taken to load the image1: %lf seconds", elapsed_time);


	start = clock();
	decodeImage(right, &image2, &width, &height);
	resizedWidth = width / 4;
	resizedHeight = height / 4;
	end = clock();
	elapsed_time = (end-start)/(double)CLOCKS_PER_SEC;
	printf("cpu time taken to load the image2: %lf seconds", elapsed_time);


	/* Host/device data structures */
	cl_platform_id platform;
	cl_device_id dev;
	cl_int err;
	cl_context context;

	/* Extension data */
	char name_data[48], ext_data[4096], vendor_data[192], driver_version[512], highest_version[512], device_version[512];
	cl_ulong global_mem_size;
	cl_uint address_bits;
	cl_bool device_available, compiler_available;
	cl_uint char_width;
	cl_uint max_compute_units, max_work_item_dim;
	cl_bool img_support;

	/* Identify a platform */
	err = clGetPlatformIDs(1, &platform, NULL);			
	if(err < 0) {			
		perror("Couldn't find any platforms");
		return EXIT_FAILURE;
	}

	/* Access a device, preferably a GPU */
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
	}
	if(err < 0) {
		perror("Couldn't access any devices");
		return EXIT_FAILURE;   
	}

	/* Access device name */
	err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 		
		48 * sizeof(char), name_data, NULL);			
	if(err < 0) {		
		perror("Couldn't read device name");
		return EXIT_FAILURE;
	}

	/* Access device vendor */
	err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 		
		192 * sizeof(char), vendor_data, NULL);			
	if(err < 0) {		
		perror("Couldn't read device vendor");
		return EXIT_FAILURE;
	}

	/* Access device extensions */
	err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 		
		4096 * sizeof(char), ext_data, NULL);			
	if(err < 0) {		
		perror("Couldn't read device extensions");
		return EXIT_FAILURE;
	}

	/* Access device global memory size */
	err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, 		
		sizeof(cl_ulong), &global_mem_size, NULL);			
	if(err < 0) {		
		perror("Couldn't read device global memory size");
		return EXIT_FAILURE;
	}

	/* Access device address size */
	err = clGetDeviceInfo(dev, CL_DEVICE_ADDRESS_BITS, 		
		sizeof(cl_uint), &address_bits, NULL);			
	if(err < 0) {		
		perror("Couldn't read device address size");
		return EXIT_FAILURE;
	}

	/* Check if device is available */
	err = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, 		
		sizeof(cl_bool), &device_available, NULL);			
	if(err < 0) {		
		perror("Couldn't check if device is available");
		return EXIT_FAILURE;
	}

	/* Check if implementation provides a compiler for the device */
	err = clGetDeviceInfo(dev, CL_DEVICE_COMPILER_AVAILABLE, 		
		sizeof(cl_bool), &compiler_available, NULL);			
	if(err < 0) {		
		perror("Couldn't check if implementation provides a compiler for the device");
		return EXIT_FAILURE;
	}

	/* Check device image support */
	err = clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, 		
		sizeof(cl_bool), &img_support, NULL);			
	if(err < 0) {		
		perror("Couldn't check device opencl version");
		return EXIT_FAILURE;
	}

	/* Access device preferred vector width in chars */
	err = clGetDeviceInfo(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, 		
		sizeof(char_width), &char_width, NULL);			
	if(err < 0) {		
		perror("Couldn't check device preferred vector width");
		return EXIT_FAILURE;
	}

	/* Check device max compute units */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, 		
		sizeof(max_compute_units), &max_compute_units, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max compute units");
		return EXIT_FAILURE;
	}

	/* Check device max work item dimensions */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 		
		sizeof(max_work_item_dim), &max_work_item_dim, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max work item dimensions");
		return EXIT_FAILURE;
	}

	/* Check device driver version */
	err = clGetDeviceInfo(dev, CL_DRIVER_VERSION, 		
		512 * sizeof(char), driver_version, NULL);			
	if(err < 0) {		
		perror("Couldn't check device driver version");
		return EXIT_FAILURE;
	}

	/* Check highest supported opencl version */
	err = clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, 		
		512 * sizeof(char), highest_version, NULL);			
	if(err < 0) {		
		perror("Couldn't check highest supported opencl version");
		return EXIT_FAILURE;
	}

	/* Check device opencl version */
	err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, 		
		512 * sizeof(char), device_version, NULL);			
	if(err < 0) {		
		perror("Couldn't check device opencl version");
		return EXIT_FAILURE;
	}

	printf("\n------------------------------------------\nDEVICE INFORMATION\n\n");
	printf("NAME: %s\nVENDOR: %s\n\nEXTENSIONS: %s\n\n", name_data, vendor_data, ext_data);
	printf("GLOBAL MEM SIZE: %lu bytes\nADDRESS BITS: %u\n", global_mem_size, address_bits);
	printf("DEVICE AVAILABLE: %d\nCOMPILER AVAILABLE: %d\n", device_available, compiler_available);
	printf("PREFERRED VECTOR WIDTH: %u chars\nMAX COMPUTE UNITS: %u\nMAX WORK ITEM DIMENSIONS: %u\n", char_width, max_compute_units, max_work_item_dim);
	printf("HIGHEST SUPPORTED OPENCL VERSION: %s\nDEVICE OPENCL VERSION: %s\n", highest_version, device_version);
	printf("CL_DEVICE_IMAGE_SUPPORT: %d\n", img_support);
	printf("\n------------------------------------------\n");


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
	const char *file_name[] = {PROGRAM_FILE_0, PROGRAM_FILE_1, PROGRAM_FILE_2};
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

	// allocate memory
	resized1 = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	resized2 = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	grayscale1 = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	grayscale2 = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);
	output = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight*4);

	// Create command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, dev, 0, &err);
	if(err < 0) {
		perror("Couldn't create command queue");
		return EXIT_FAILURE;   
	}

	// Create memory buffers on the device
	cl_mem input_clmem1 = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem input_clmem2 = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem1 = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem resized_clmem2 = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("0 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem1 = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("1 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem grayscale_clmem2 = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("1 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}
	cl_mem output_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
	if(err < 0) {
		perror("2 Couldn't create memory buffers on the device");
		return EXIT_FAILURE;   
	}

	// Copy buffers to the device
	err = clEnqueueWriteBuffer(command_queue, input_clmem1, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, input_clmem2, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, resized_clmem1, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, resized_clmem2, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, grayscale_clmem1, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), grayscale1, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, grayscale_clmem2, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), grayscale2, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, output_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), output, 0, NULL, NULL);
	
	if(err < 0) {
		perror("Couldn't copy memory buffers to the device");
		return EXIT_FAILURE;   
	}

	// Create kernel
	cl_kernel kernel0 = clCreateKernel(program, KERNEL_NAME_0, &err);
	if(err < 0) {
		perror("0 Couldn't create kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel kernel00 = clCreateKernel(program, KERNEL_NAME_0, &err);
	if(err < 0) {
		perror("00 Couldn't create kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel kernel1 = clCreateKernel(program, KERNEL_NAME_1, &err);
	if(err < 0) {
		perror("1 Couldn't create kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel kernel2 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		perror("2 Couldn't create kernel");
		return EXIT_FAILURE;   
	}
	cl_kernel kernel22 = clCreateKernel(program, KERNEL_NAME_2, &err);
	if(err < 0) {
		perror("22 Couldn't create kernel");
		return EXIT_FAILURE;   
	}

	// Set kernel arguments
	err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&input_clmem1);
	if(err < 0) {
		perror("0 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&resized_clmem1);
	if(err < 0) {
		perror("1 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel22, 0, sizeof(cl_mem), (void*)&input_clmem2);
	if(err < 0) {
		perror("00 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel22, 1, sizeof(cl_mem), (void*)&resized_clmem2);
	if(err < 0) {
		perror("11 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}

	err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void*)&resized_clmem1);
	if(err < 0) {
		perror("2 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void*)&grayscale_clmem1);
	if(err < 0) {
		perror("3 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel00, 0, sizeof(cl_mem), (void*)&resized_clmem2);
	if(err < 0) {
		perror("22 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel00, 1, sizeof(cl_mem), (void*)&grayscale_clmem2);
	if(err < 0) {
		perror("33 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}


	err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&grayscale_clmem1);
	if(err < 0) {
		perror("4 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&grayscale_clmem2);
	if(err < 0) {
		perror("4 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}

	err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void*)&output_clmem);
	if(err < 0) {
		perror("5 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel1, 3, sizeof(unsigned int), &resizedWidth);
	if(err < 0) {
		perror("6 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel1, 4, sizeof(unsigned int), &resizedHeight);
	if(err < 0) {
		perror("7 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel1, 5, sizeof(unsigned int), &B);
	if(err < 0) {
		perror("7 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel1, 6, sizeof(unsigned int), &MIN_DISPARITY);
	if(err < 0) {
		perror("7 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	err = clSetKernelArg(kernel1, 7, sizeof(unsigned int), &MAX_DISPARITY);
	if(err < 0) {
		perror("7 Couldn't set kernel arguments");
		return EXIT_FAILURE;   
	}
	


	// Execute kernel on the device
	size_t global_size[2] = {(size_t)width*4, (size_t)height*4};
	size_t global_work_offset[2] = {0, 0};
	size_t global_size_resized = resizedWidth*resizedHeight*4;
	cl_event event_list[2];

	// Resize
	err = clEnqueueNDRangeKernel(command_queue, kernel2, 2, global_work_offset, global_size, NULL, 0, NULL, &event_list[0]);
	if(err < 0) {
		perror("0 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, kernel22, 2, global_work_offset, global_size, NULL, 0, NULL, &event_list[0]);
	if(err < 0) {
		perror("0 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// Grayscale
	err = clEnqueueNDRangeKernel(command_queue, kernel0, 1, NULL, &global_size_resized, NULL, 1, event_list, &event_list[1]);
	if(err < 0) {
		perror("1 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	err = clEnqueueNDRangeKernel(command_queue, kernel00, 1, NULL, &global_size_resized, NULL, 1, event_list, &event_list[1]);
	if(err < 0) {
		perror("1 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}

	// ZNCC filter
	err = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global_size_resized, NULL, 2, event_list, NULL);
	if(err < 0) {
		perror("2 Error in clEnqueueNDRangeKernel");
		return EXIT_FAILURE;   
	}
	
	// Read results
	err = clEnqueueReadBuffer(command_queue, output_clmem, CL_TRUE, 0, resizedWidth*resizedHeight*4*sizeof(unsigned char), output, 0, NULL, NULL);
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
	encodeImage(outputimg, output, &resizedWidth, &resizedHeight);

	end = clock();
	elapsed_time = (end-start)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken to encode image: %lf seconds", elapsed_time);


	/* Deallocate resources */
	for(unsigned i = 0; i < NUM_FILES; i++) {
		free(program_buffer[i]);
	}
	err = clReleaseKernel(kernel0);
	err = clReleaseKernel(kernel00);
	err = clReleaseKernel(kernel1);
	err = clReleaseKernel(kernel2);
	err = clReleaseKernel(kernel22);
	err = clReleaseProgram(program);
	err = clReleaseMemObject(input_clmem1);
	err = clReleaseMemObject(input_clmem2);
	err = clReleaseMemObject(grayscale_clmem1);
	err = clReleaseMemObject(grayscale_clmem2);
	err = clReleaseMemObject(resized_clmem1);
	err = clReleaseMemObject(resized_clmem2);
	err = clReleaseMemObject(output_clmem);
	err = clReleaseCommandQueue(command_queue);
	err = clReleaseContext(context);
	if(err < 0) {
		perror("Error deallocating resources");
		return EXIT_FAILURE;   
	}
	free(image1);
	free(image2);
	free(resized1);
	free(resized2);
	free(grayscale1);
	free(grayscale2);
	free(output);


	printf("\n\nProgram finished\n");
	return EXIT_SUCCESS;
	}
