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

   /* Extension data */
   char name_data[48], ext_data[4096], vendor_data[192];
   cl_ulong global_mem_size;
   cl_uint address_bits;
   cl_bool device_available, compiler_available;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);			
   if(err < 0) {			
      perror("Couldn't find any platforms");
      EXIT_FAILURE;
   }

   /* Access a device, preferably a GPU */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      EXIT_FAILURE;   
   }

   /* Access device name */
   err = clGetDeviceInfo(dev, CL_DEVICE_NAME, 		
      48 * sizeof(char), name_data, NULL);			
   if(err < 0) {		
      perror("Couldn't read device name");
      EXIT_FAILURE;
   }

   /* Access device vendor */
   err = clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 		
      192 * sizeof(char), vendor_data, NULL);			
   if(err < 0) {		
      perror("Couldn't read device vendor");
      EXIT_FAILURE;
   }

   /* Access device extensions */
   err = clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 		
      4096 * sizeof(char), ext_data, NULL);			
   if(err < 0) {		
      perror("Couldn't read device extensions");
      EXIT_FAILURE;
   }

   /* Access device global memory size */
   err = clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, 		
      sizeof(cl_ulong), &global_mem_size, NULL);			
   if(err < 0) {		
      perror("Couldn't read device global memory size");
      EXIT_FAILURE;
   }

   /* Access device address size */
   err = clGetDeviceInfo(dev, CL_DEVICE_ADDRESS_BITS, 		
      sizeof(cl_uint), &address_bits, NULL);			
   if(err < 0) {		
      perror("Couldn't read device address size");
      EXIT_FAILURE;
   }

   /* Check if device is available */
   err = clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, 		
      sizeof(cl_bool), &device_available, NULL);			
   if(err < 0) {		
      perror("Couldn't check if device is available");
      EXIT_FAILURE;
   }

   /* Check if implementation provides a compiler for the device */
   err = clGetDeviceInfo(dev, CL_DEVICE_COMPILER_AVAILABLE, 		
      sizeof(cl_bool), &compiler_available, NULL);			
   if(err < 0) {		
      perror("Couldn't check if implementation provides a compiler for the device");
      EXIT_FAILURE;
   }

   printf("\nDEVICE INFORMATION\n\n");
   printf("NAME: %s\nVENDOR: %s\nEXTENSIONS: %s\n\n", name_data, vendor_data, ext_data);
   printf("GLOBAL MEM SIZE: %lu\nADDRESS BITS: %u\n", global_mem_size, address_bits);
   printf("DEVICE AVAILABLE: %d\nCOMPILER AVAILABLE: %d\n", device_available, compiler_available);

   printf("\nProgram finished\n");
   return EXIT_SUCCESS;
}