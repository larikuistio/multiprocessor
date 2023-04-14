
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

	/* Variables to store device information on */
	char name_data[48], ext_data[4096], vendor_data[192], driver_version[512], highest_version[512], device_version[512];
	cl_ulong global_mem_size;
	cl_uint address_bits;
	cl_bool device_available, compiler_available;
	cl_uint char_width;
	cl_uint max_compute_units, max_work_item_dim;
	cl_bool img_support;
	cl_device_local_mem_type local_mem_type;
	cl_ulong local_mem_size;
	cl_uint max_clock_frequency;
	cl_ulong max_constant_buffer_size;
	size_t max_work_group_size;

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

	/* Check device local memory type */
	err = clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_TYPE, 		
		sizeof(local_mem_type), &local_mem_type, NULL);			
	if(err < 0) {		
		perror("Couldn't check local memory type");
		return EXIT_FAILURE;
	}

	/* Check device local memory size */
	err = clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, 		
		sizeof(local_mem_size), &local_mem_size, NULL);			
	if(err < 0) {		
		perror("Couldn't check device local memory size");
		return EXIT_FAILURE;
	}

	/*  Check device max clock frequency */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, 		
		sizeof(max_clock_frequency), &max_clock_frequency, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max clock frequency");
		return EXIT_FAILURE;
	}

	/* Check device max constant buffer size */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
		sizeof(max_constant_buffer_size), &max_constant_buffer_size, NULL);			
	if(err < 0) {		
		perror("Couldn't check device  max constant buffer size");
		return EXIT_FAILURE;
	}

	/* Check device max work group size */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,	
		sizeof(max_work_group_size), &max_work_group_size, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max work group size");
		return EXIT_FAILURE;
	}

	printf("\n------------------------------------------\nDEVICE INFORMATION\n\n");
	printf("\nNAME: %s\nVENDOR: %s\n\nEXTENSIONS: %s\n\n", name_data, vendor_data, ext_data);
	printf("\nGLOBAL MEM SIZE: %lu bytes\nADDRESS BITS: %u", global_mem_size, address_bits);
	printf("\nDEVICE AVAILABLE: %d\nCOMPILER AVAILABLE: %d", device_available, compiler_available);
	printf("\nPREFERRED VECTOR WIDTH: %u chars\nMAX COMPUTE UNITS: %u\nMAX WORK ITEM DIMENSIONS: %u", char_width, max_compute_units, max_work_item_dim);
	printf("\nHIGHEST SUPPORTED OPENCL VERSION: %s\nDEVICE OPENCL VERSION: %s", highest_version, device_version);
	printf("\nDEVICE_IMAGE_SUPPORT: %d", img_support);
	printf("\nDEVICE_LOCAL_MEM_TYPE: %d (1=local, 2=global)", local_mem_type);
	printf("\nDEVICE_LOCAL_MEM_SIZE: %d", local_mem_size);
	printf("\nDEVICE_MAX_CLOCK_FREQUENCY: %d", max_clock_frequency);
	printf("\nDEVICE_MAX_CONSTANT_BUFFER_SIZE: %d", max_constant_buffer_size);
	printf("\nDEVICE_MAX_WORK_GROUP_SIZE: %d", max_work_group_size);
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
