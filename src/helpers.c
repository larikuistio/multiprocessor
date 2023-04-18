#include "helpers.h"

#define PIXEL_SIZE 4
#define SUPERMAGICNUMBER 3

void decodeImage(const char* filename, unsigned char** image, unsigned* width, unsigned* height) {

	unsigned error = 0;
	unsigned char* png = 0;
	size_t pngsize;

	error = lodepng_load_file(&png, &pngsize, filename);
	if(!error) error = lodepng_decode32(image, width, height, png, pngsize);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(png);
}

void encodeImage(const char* filename, unsigned char* image, unsigned* width, unsigned* height) {

	unsigned error = 0;
	unsigned char* png = 0;
	size_t pngsize;
    
	error = lodepng_encode32(&png, &pngsize, image, *width, *height);
	if(!error) lodepng_save_file(png, pngsize, filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(png);

}

bool convertToGrayscale(unsigned char* image, unsigned char** newimage, unsigned *width, unsigned *height) {

    *newimage = malloc((*width) * (*height) * sizeof(unsigned char));
    
    if (*newimage == NULL) {
        printf("malloc failed\n");
        return false;
    }
    
    int j = 0;
    for(unsigned i = 0; i < (*width) * (*height) * PIXEL_SIZE; i += PIXEL_SIZE)
    {
        (*newimage)[j] = 0.2126*image[i] + 0.7152*image[i+1] + 0.0722*image[i+2];
        j++;
    }
    return true;
}

bool convertToRGB(unsigned char* image, unsigned char** newimage, unsigned *width, unsigned *height) {

    *newimage = malloc((*width) * (*height) * 4 * sizeof(unsigned char));
    
    if (*newimage == NULL) {
        printf("malloc failed\n");
        return false;
    }
    
    int j = 0;
    for(unsigned i = 0; i < (*width) * (*height); i++)
    {
        (*newimage)[j] = image[i];
        j++;
        (*newimage)[j] = image[i];
        j++;
        (*newimage)[j] = image[i];
        j++;
        (*newimage)[j] = 255;
        j++;
    }
    return true;
}

bool resizeImage(unsigned char* image, unsigned char** newimage, unsigned* width, unsigned* height, unsigned* newwidth, unsigned* newheight) {
    
    *newwidth = (unsigned)*width / 4;
    *newheight = (unsigned)*height / 4;
    *newimage = malloc((*newwidth) * (*newheight) * PIXEL_SIZE);

    if (*newimage == NULL) {
        printf("malloc failed\n");
        return false;
    }
    
    int j = 0;
    for(unsigned i = 0; i < (*width) * (*height-SUPERMAGICNUMBER) * PIXEL_SIZE; i += PIXEL_SIZE*4)
    {
        // printf("%u\n", i);
        if(i % ((*width) * PIXEL_SIZE) == 0 && i != 0)
        {
            i += (*width)* PIXEL_SIZE * 3;
        }
        (*newimage)[j]     = image[i];
        (*newimage)[j + 1] = image[i + 1];
        (*newimage)[j + 2] = image[i + 2];
        (*newimage)[j + 3] = image[i + 3];
        j += 4;

    }
    return true;
}

bool addMatrix(unsigned** in_a, unsigned** in_b, unsigned** out, unsigned row_size, unsigned col_size) {
    
    unsigned i, j;
    for( i = 0; i < row_size; i++ ) {
        for( j = 0 ; j < col_size; j++ ) {
                out[i][j] = in_a[i][j] + in_b[i][j];    
        }
    }
    return EXIT_SUCCESS;
}


int printDeviceInfo(cl_device_id *device, cl_platform_id *plat)
{
	cl_platform_id platform = *plat;
	cl_device_id dev = *device;
	cl_int err;

	/* Extension data */
	char name_data[48], ext_data[4096], vendor_data[192], driver_version[512], highest_version[512], device_version[512];
	cl_ulong global_mem_size;
	cl_uint address_bits;
	cl_bool device_available, compiler_available;
	cl_uint char_width;
	cl_uint max_compute_units, max_work_item_dim;
	cl_bool img_support;
	cl_device_local_mem_type local_mem_type;
	cl_ulong local_mem_size, max_constant_buffer_size;
	cl_uint max_clock_frequency;
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

	/* Check device local mem type */
	err = clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_TYPE, 		
		sizeof(cl_device_local_mem_type), &local_mem_type, NULL);			
	if(err < 0) {		
		perror("Couldn't check device local mem type");
		return EXIT_FAILURE;
	}

	/* Check device local mem size */
	err = clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, 		
		sizeof(cl_ulong), &local_mem_size, NULL);			
	if(err < 0) {		
		perror("Couldn't check device local mem size");
		return EXIT_FAILURE;
	}

	/* Check device max clock frequency */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, 		
		sizeof(cl_uint), &max_clock_frequency, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max clock frequency");
		return EXIT_FAILURE;
	}

	/* Check device max constant buffer size */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 		
		sizeof(cl_ulong), &max_constant_buffer_size, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max constant buffer size");
		return EXIT_FAILURE;
	}

	/* Check device max constant buffer size */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, 		
		sizeof(size_t), &max_work_group_size, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max work group size");
		return EXIT_FAILURE;
	}

	size_t max_work_item_sizes[max_work_item_dim];
	/* Check device max constant buffer size */
	err = clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, 		
		max_work_item_dim * sizeof(size_t), &max_work_item_sizes, NULL);			
	if(err < 0) {		
		perror("Couldn't check device max work item sizes");
		return EXIT_FAILURE;
	}

	printf("\n------------------------------------------\nDEVICE INFORMATION\n\n");
	printf("NAME: %s\nVENDOR: %s\n\nEXTENSIONS: %s\n\n", name_data, vendor_data, ext_data);
	printf("GLOBAL MEM SIZE: %lu bytes\nADDRESS BITS: %u\n", global_mem_size, address_bits);
	printf("DEVICE AVAILABLE: %d\nCOMPILER AVAILABLE: %d\n", device_available, compiler_available);
	printf("PREFERRED VECTOR WIDTH: %u chars\nMAX COMPUTE UNITS: %u\nMAX WORK ITEM DIMENSIONS: %u\n", char_width, max_compute_units, max_work_item_dim);
	printf("HIGHEST SUPPORTED OPENCL VERSION: %s\nDEVICE OPENCL VERSION: %s\n", highest_version, device_version);
	printf("CL_DEVICE_IMAGE_SUPPORT: %d\n", img_support);
	if(local_mem_type == CL_LOCAL)
	{
		printf("CL_DEVICE_LOCAL_MEM_TYPE: CL_LOCAL\n");
	}
	else if(local_mem_type == CL_GLOBAL)
	{
		printf("CL_DEVICE_LOCAL_MEM_TYPE: CL_GLOBAL\n");
	}
	else if(local_mem_type == CL_NONE)
	{
		printf("CL_DEVICE_LOCAL_MEM_TYPE: CL_NONE\n");
	}
	else
	{
		printf("CL_DEVICE_LOCAL_MEM_TYPE: UNKNOWN\n");
	}
	printf("CL_DEVICE_LOCAL_MEM_SIZE: %lu bytes\n", local_mem_size);
	printf("CL_DEVICE_MAX_CLOCK_FREQUENCY: %lu MHz\n", max_clock_frequency);
	printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %lu bytes\n", max_constant_buffer_size);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", max_work_group_size);
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES: %lu\n", max_work_item_sizes);
	
	printf("");
	printf("\n------------------------------------------\n");

    *device = dev;
    *plat = platform;

	return EXIT_SUCCESS;
}