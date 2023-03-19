#define NUM_FILES 1
#define PROGRAM_FILE_0 "kernels/addmatrix.cl"
#define KERNEL_NAME "addmatrix"
#define MATRIX_SIZE 10

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>
#include <CL/cl.h>
#include "helpers.h"

int add_matrices(void) {

    unsigned array_row_size = 10;
    unsigned array_col_size = 10;

    unsigned** arr1 = calloc(array_row_size, sizeof(unsigned*));
    unsigned** arr2 = calloc(array_row_size, sizeof(unsigned*));
    unsigned** added_arr = calloc(array_row_size, sizeof(unsigned*));


    unsigned i, j, q;
    for( i = 0, q = 0 ; i < array_row_size; i++ ) {
        arr1[i] =       calloc(array_col_size, sizeof(unsigned*));
        arr2[i] =       calloc(array_col_size, sizeof(unsigned*));
        added_arr[i] =  calloc(array_col_size, sizeof(unsigned*));
        for( j = 0 ; j < array_col_size; j++ ) {
            arr1[i][j] = ++q;
            arr2[i][j] = ++q;
        }
    }

    addMatrix(arr1, arr2, added_arr, array_row_size, array_col_size);


    for( i = 0; i < array_row_size; i++ ) {
        free(arr1[i]);
        free(arr2[i]);
        free(added_arr[i]);
    }

    free(arr1);
    free(arr2);
    free(added_arr);
    
    return EXIT_SUCCESS;
}

int main(void) {

    if(add_matrices()) {
        printf("Add matrices failed\n");
        return EXIT_FAILURE;
    
    }

   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id dev;
   cl_int err;
   cl_context context;

   /* Extension data */
   char name_data[48], ext_data[4096], vendor_data[192];
   cl_ulong global_mem_size;
   cl_uint address_bits;
   cl_bool device_available, compiler_available;
   cl_uint char_width;
   cl_uint max_compute_units;

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

   printf("\n------------------------------------------\nDEVICE INFORMATION\n\n");
   printf("NAME: %s\nVENDOR: %s\n\nEXTENSIONS: %s\n\n", name_data, vendor_data, ext_data);
   printf("GLOBAL MEM SIZE: %lu bytes\nADDRESS BITS: %u\n", global_mem_size, address_bits);
   printf("DEVICE AVAILABLE: %d\nCOMPILER AVAILABLE: %d\n", device_available, compiler_available);
   printf("PREFERRED VECTOR WIDTH: %u chars\nMAX COMPUTE UNITS: %u\n", char_width, max_compute_units);
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
   const char *file_name[] = {PROGRAM_FILE_0};
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

   unsigned x = 1;
   // Create matrixes
   int *A = (int*)malloc(sizeof(int)*MATRIX_SIZE);
   int *B = (int*)malloc(sizeof(int)*MATRIX_SIZE);
   int *results = (int*)malloc(sizeof(int)*MATRIX_SIZE);
   for(unsigned i = 0; i < MATRIX_SIZE; i++) {
      A[i] = x;
      B[i] = x * 2;
      results[i] = 0;
      x++;
   }
   
   

   // Create command queue
   cl_command_queue command_queue = clCreateCommandQueue(context, dev, 0, &err);
   if(err < 0) {
      perror("Couldn't create command queue");
      return EXIT_FAILURE;   
   }

   // Create memory buffers on the device
   cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * sizeof(int), NULL, &err);
   cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * sizeof(int), NULL, &err);
   cl_mem results_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 4 * sizeof(int), NULL, &err);
   if(err < 0) {
      perror("Couldn't create memory buffers on the device");
      return EXIT_FAILURE;   
   }

   // Copy buffers to the device
   err = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, MATRIX_SIZE * sizeof(int), A, 0, NULL, NULL);
   err = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, MATRIX_SIZE * sizeof(int), B, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't copy memory buffers to the device");
      return EXIT_FAILURE;   
   }

   // Create kernel
   cl_kernel kernel = clCreateKernel(program, "add_matrix", &err);
   if(err < 0) {
      perror("Couldn't create kernel");
      return EXIT_FAILURE;   
   }

   // Set kernel arguments
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&A_clmem);
   err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&B_clmem);
   err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&results_clmem);
   if(err < 0) {
      perror("Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }

   // Execute kernel on the device
   size_t global_size = MATRIX_SIZE;
   size_t local_size = 10;
   err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
   if(err < 0) {
      perror("Error in clEnqueueNDRangeKernel");
      return EXIT_FAILURE;   
   }
   
   // Read results
   err = clEnqueueReadBuffer(command_queue, results_clmem, CL_TRUE, 0, MATRIX_SIZE*sizeof(int), results, 0, NULL, NULL);
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
   
   // print results
   for(unsigned i = 0; i < MATRIX_SIZE; i++) {
      printf(" %d ", results[i]);
   }


   /* Deallocate resources */
   for(unsigned i = 0; i < NUM_FILES; i++) {
      free(program_buffer[i]);
   }
   err = clReleaseKernel(kernel);
   err = clReleaseProgram(program);
   err = clReleaseMemObject(A_clmem);
   err = clReleaseMemObject(B_clmem);
   err = clReleaseMemObject(results_clmem);
   err = clReleaseCommandQueue(command_queue);
   err = clReleaseContext(context);
   if(err < 0) {
      perror("Error deallocating resources");
      return EXIT_FAILURE;   
   }
   free(A);
   free(B);
   free(results);


   printf("\n\nProgram finished\n");
   return EXIT_SUCCESS;
}