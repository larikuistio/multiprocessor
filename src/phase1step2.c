#define NUM_FILES 1
#define PROGRAM_FILE_0 "kernels/addmatrix.cl"
#define KERNEL_NAME "add_matrix"
#define MATRIX_SIZE 10000

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <CL/cl.h>
#include "helpers.h"
#include <time.h>
#include <sys/time.h>

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

int main(int argc, char **argv) {

   clock_t startprogclk = clock();
	double startprog = queryProfiler();

   clock_t startclk = clock();
   double start = queryProfiler();
   if(add_matrices()) {
      printf("Add matrices failed\n");
      return EXIT_FAILURE;
   
   }
   double end = queryProfiler();
   clock_t endclk = clock();
   double elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
   printf("\ncpu time taken to add matrices: %lf seconds\n", elapsed_time);
   printf("real time taken to add matrices: %lf seconds\n", end-start);

   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id dev;
   cl_int err;
   cl_context context;

   cl_ulong kernelstarttimes[3];
	cl_ulong kernelendtimes[3];
	cl_double exectimems[3];

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
   cl_command_queue command_queue = clCreateCommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create command queue");
      return EXIT_FAILURE;   
   }

   // Create memory buffers on the device
   cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * sizeof(int), NULL, &err);
   cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * sizeof(int), NULL, &err);
   cl_mem results_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * sizeof(int), NULL, &err);
   if(err < 0) {
      perror("Couldn't create memory buffers on the device");
      return EXIT_FAILURE;   
   }

   cl_event event_list[4];

   // Copy buffers to the device
   err = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, MATRIX_SIZE * sizeof(int), A, 0, NULL, &event_list[0]);
   err = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, MATRIX_SIZE * sizeof(int), B, 0, NULL, &event_list[1]);
   if(err < 0) {
      perror("Couldn't copy memory buffers to the device");
      return EXIT_FAILURE;   
   }

   // Create kernel
   cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
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
   err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 2, event_list, &event_list[2]);
   if(err < 0) {
      perror("Error in clEnqueueNDRangeKernel");
      return EXIT_FAILURE;   
   }
   
   // Read results
   err = clEnqueueReadBuffer(command_queue, results_clmem, CL_TRUE, 0, MATRIX_SIZE*sizeof(int), results, 3, event_list, &event_list[3]);
   if(err < 0) {
      perror("Error in clEnqueueReadBuffer");
      return EXIT_FAILURE;   
   }


   // profile kernel execution times
	clWaitForEvents(4, (const cl_event*)&event_list);

   // write buffer
	clGetEventProfilingInfo(event_list[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[0], NULL);
	clGetEventProfilingInfo(event_list[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[0], NULL);
	exectimems[0] = (cl_double)(kernelendtimes[0] - kernelstarttimes[0])*(cl_double)(1e-06);
	printf("writing input buffers to device took: %lf ms\n", exectimems[0]);

	// matrix addition
	clGetEventProfilingInfo(event_list[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[1], NULL);
	clGetEventProfilingInfo(event_list[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[1], NULL);
	exectimems[1] = (cl_double)(kernelendtimes[1] - kernelstarttimes[1])*(cl_double)(1e-06);
	printf("matrix addition kernel execution time: %lf ms\n", exectimems[1]);

   // read buffer
	clGetEventProfilingInfo(event_list[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[2], NULL);
	clGetEventProfilingInfo(event_list[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[2], NULL);
	exectimems[2] = (cl_double)(kernelendtimes[2] - kernelstarttimes[2])*(cl_double)(1e-06);
	printf("reading output buffer from device took: %lf ms\n", exectimems[2]);


   // Clean up and wait for all the commands to complete
   err = clFlush(command_queue);
   err = clFinish(command_queue);
   if(err < 0) {
      perror("Error executing command queue");
      return EXIT_FAILURE;   
   }
   
   if (argc > 1) {
      if (strcmp(argv[1], "print-results") == 0) {
         // print results
         for(unsigned i = 0; i < MATRIX_SIZE; i++) {
            printf(" %d ", results[i]);
         }
      }
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

   clock_t endprogclk = clock();
	double endprog = queryProfiler();

	double elapsed_time_prog = (endprogclk-startprogclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken by program execution: %lf seconds\n", elapsed_time_prog);
	printf("real time taken by program execution: %f  seconds\n", endprog-startprog);

   return EXIT_SUCCESS;
}