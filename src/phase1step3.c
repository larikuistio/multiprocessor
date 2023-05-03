#define NUM_FILES 3
#define PROGRAM_FILE_0 "kernels/movingfilter5x5.cl"
#define PROGRAM_FILE_1 "kernels/grayscalergba.cl"
#define PROGRAM_FILE_2 "kernels/resizeimage.cl"
#define KERNEL_NAME_0 "grayscale"
#define KERNEL_NAME_1 "movingfilter5x5"
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
#include <sys/time.h>

/*
Some parts are based on examples provided in this book:
	
Matthew Scarpino
OpenCL in action : how to accelerate graphics and computation

https://oula.finna.fi/Record/oy.9917467213906252?sid=2954376879
*/

int main(int argc, char **argv) {

   // profile entire program execution time
   clock_t startprogclk = clock();
	double startprog = queryProfiler();

   // exit if required file names not provided
   if (argc < 2) {
      printf("provide input and output image files names as arguments\n");
      return EXIT_FAILURE;
   }

   const char* inputimg = argv[1];
   const char* outputimg = argv[2];
   unsigned char* image = 0;
   unsigned char* grayscale = 0;
   unsigned char* output = 0;
   unsigned char* resized = 0;
   unsigned width, height, resizedWidth, resizedHeight;

   clock_t startclk = clock();
   double start = queryProfiler();
   // read input image from file
   decodeImage(inputimg, &image, &width, &height);
   resizedWidth = width / 4;
   resizedHeight = height / 4;
   double end = queryProfiler();
   clock_t endclk = clock();

   double elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
   printf("\ncpu time taken to load the image: %lf seconds\n", elapsed_time);
   printf("real time taken to load the image: %lf seconds\n", end-start);

   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id dev;
   cl_int err;
   cl_context context;

   cl_ulong kernelstarttimes[5];
	cl_ulong kernelendtimes[5];
	cl_double exectimems[5];

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
   output = (unsigned char*)malloc(sizeof(unsigned char)*resizedWidth*resizedHeight);

   // Create command queue
   cl_command_queue command_queue = clCreateCommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create command queue");
      return EXIT_FAILURE;   
   }

   // Create memory buffers on the device
   cl_mem input_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), NULL, &err);
   if(err < 0) {
      perror("0 Couldn't create memory buffers on the device");
      return EXIT_FAILURE;   
   }
   cl_mem resized_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), NULL, &err);
   if(err < 0) {
      perror("1 Couldn't create memory buffers on the device");
      return EXIT_FAILURE;   
   }
   cl_mem grayscale_clmem = clCreateBuffer(context, CL_MEM_READ_WRITE, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
   if(err < 0) {
      perror("2 Couldn't create memory buffers on the device");
      return EXIT_FAILURE;   
   }
   cl_mem output_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resizedWidth * resizedHeight * sizeof(unsigned char), NULL, &err);
   if(err < 0) {
      perror("3 Couldn't create memory buffers on the device");
      return EXIT_FAILURE;   
   }

   cl_event event_list[5];

   // Copy buffers to the device
   err = clEnqueueWriteBuffer(command_queue, input_clmem, CL_TRUE, 0, width * height * 4 * sizeof(unsigned char), image, 0, event_list, &event_list[0]);
   err = clEnqueueWriteBuffer(command_queue, resized_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * 4 * sizeof(unsigned char), resized, 0, NULL, NULL);
   err = clEnqueueWriteBuffer(command_queue, grayscale_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), grayscale, 0, NULL, NULL);
   err = clEnqueueWriteBuffer(command_queue, output_clmem, CL_TRUE, 0, resizedWidth * resizedHeight * sizeof(unsigned char), output, 0, NULL, NULL);
   
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

   // Set kernel arguments
   err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void*)&input_clmem);
   if(err < 0) {
      perror("0 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void*)&resized_clmem);
   if(err < 0) {
      perror("1 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void*)&resized_clmem);
   if(err < 0) {
      perror("2 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void*)&grayscale_clmem);
   if(err < 0) {
      perror("3 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void*)&grayscale_clmem);
   if(err < 0) {
      perror("4 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void*)&output_clmem);
   if(err < 0) {
      perror("5 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel1, 2, sizeof(unsigned int), &resizedWidth);
   if(err < 0) {
      perror("6 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }
   err = clSetKernelArg(kernel1, 3, sizeof(unsigned int), &resizedHeight);
   if(err < 0) {
      perror("7 Couldn't set kernel arguments");
      return EXIT_FAILURE;   
   }


   // Execute kernel on the device
   size_t global_size[2] = {(size_t)width*4, (size_t)height*4};
   size_t global_work_offset[2] = {0, 0};
   size_t global_size_resized = resizedWidth*resizedHeight*4;
   

   // Resize
   err = clEnqueueNDRangeKernel(command_queue, kernel2, 2, global_work_offset, global_size, NULL, 0, NULL, &event_list[1]);
   if(err < 0) {
      perror("0 Error in clEnqueueNDRangeKernel");
      return EXIT_FAILURE;   
   }

   // Grayscale
   err = clEnqueueNDRangeKernel(command_queue, kernel0, 1, NULL, &global_size_resized, NULL, 1, event_list, &event_list[2]);
   if(err < 0) {
      perror("1 Error in clEnqueueNDRangeKernel");
      return EXIT_FAILURE;   
   }

   // Moving filter
   err = clEnqueueNDRangeKernel(command_queue, kernel1, 1, NULL, &global_size_resized, NULL, 2, event_list, &event_list[3]);
   if(err < 0) {
      perror("2 Error in clEnqueueNDRangeKernel");
      return EXIT_FAILURE;   
   }
   
   // Read results
   err = clEnqueueReadBuffer(command_queue, output_clmem, CL_TRUE, 0, resizedWidth*resizedHeight*sizeof(unsigned char), output, 3, event_list, &event_list[4]);
   if(err < 0) {
      perror("Error in clEnqueueReadBuffer");
      return EXIT_FAILURE;
   }

   // profile kernel execution times
	clWaitForEvents(5, (const cl_event*)&event_list);

   // write buffer
	clGetEventProfilingInfo(event_list[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[0], NULL);
	clGetEventProfilingInfo(event_list[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[0], NULL);
	exectimems[0] = (cl_double)(kernelendtimes[0] - kernelstarttimes[0])*(cl_double)(1e-06);
	printf("writing input buffer to device took: %lf ms\n", exectimems[0]);

	// resize
	clGetEventProfilingInfo(event_list[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[1], NULL);
	clGetEventProfilingInfo(event_list[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[1], NULL);
	exectimems[1] = (cl_double)(kernelendtimes[1] - kernelstarttimes[1])*(cl_double)(1e-06);
	printf("resize kernel execution time: %lf ms\n", exectimems[1]);

	// grayscale
	clGetEventProfilingInfo(event_list[2], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[2], NULL);
	clGetEventProfilingInfo(event_list[2], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[2], NULL);
	exectimems[2] = (cl_double)(kernelendtimes[2] - kernelstarttimes[2])*(cl_double)(1e-06);
	printf("grayscale kernel execution time: %lf ms\n", exectimems[2]);

	// movingfilter
	clGetEventProfilingInfo(event_list[3], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[3], NULL);
	clGetEventProfilingInfo(event_list[3], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[3], NULL);
	exectimems[3] = (cl_double)(kernelendtimes[3] - kernelstarttimes[3])*(cl_double)(1e-06);
	printf("movingfilter kernel execution time: %lf ms\n", exectimems[3]);

   // read buffer
	clGetEventProfilingInfo(event_list[4], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelstarttimes[4], NULL);
	clGetEventProfilingInfo(event_list[4], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelendtimes[4], NULL);
	exectimems[4] = (cl_double)(kernelendtimes[4] - kernelstarttimes[4])*(cl_double)(1e-06);
	printf("reading output buffer from device took: %lf ms\n", exectimems[4]);

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
   lodepng_encode_file(outputimg, output, resizedWidth, resizedHeight, LCT_GREY, 8);
   endclk = clock();
   end = queryProfiler();
   elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
   printf("\ncpu time taken to encode image: %lf seconds\n", elapsed_time);
   printf("real time taken to encode image: %lf seconds\n", end-start);


   /* Deallocate resources */
   for(unsigned i = 0; i < NUM_FILES; i++) {
      free(program_buffer[i]);
   }
   err = clReleaseKernel(kernel0);
   err = clReleaseKernel(kernel1);
   err = clReleaseKernel(kernel2);
   err = clReleaseProgram(program);
   err = clReleaseMemObject(input_clmem);
   err = clReleaseMemObject(grayscale_clmem);
   err = clReleaseMemObject(resized_clmem);
   err = clReleaseMemObject(output_clmem);
   err = clReleaseCommandQueue(command_queue);
   err = clReleaseContext(context);
   if(err < 0) {
      perror("Error deallocating resources");
      return EXIT_FAILURE;   
   }
   free(image);
   free(output);


   printf("\n\nProgram finished\n");

   clock_t endprogclk = clock();
	double endprog = queryProfiler();

	double elapsed_time_prog = (endprogclk-startprogclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken by program execution: %lf seconds\n", elapsed_time_prog);
	printf("real time taken by program execution: %f  seconds\n", endprog-startprog);

   return EXIT_SUCCESS;
}