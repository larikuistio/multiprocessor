#include "helpers.h"

void decodeImage(const char* filename, unsigned char** image, unsigned* width, unsigned* height) {

	unsigned error = 0;
	unsigned char* png = 0;
	size_t pngsize;

	error = lodepng_load_file(&png, &pngsize, filename);
	if(!error) error = lodepng_decode24(image, width, height, png, pngsize);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(png);
}

void encodeImage(const char* filename, unsigned char* image, unsigned* width, unsigned* height) {

	unsigned error = 0;
	unsigned char* png = 0;
	size_t pngsize;
    
	error = lodepng_encode24(&png, &pngsize, image, *width, *height);
	if(!error) lodepng_save_file(png, pngsize, filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(png);

}

bool resizeImage(unsigned char* image, unsigned char** newimage, unsigned* width, unsigned* height, unsigned* newwidth, unsigned* newheight, unsigned pixel_size) {
    
    *newwidth = (unsigned)*width / 4;
    *newheight = (unsigned)*height / 4;
    *newimage = malloc((*newwidth) * (*newheight) * pixel_size * sizeof(unsigned char));

    if (*newimage == NULL) {
        printf("malloc failed\n");
        return false;
    }
    
    int j = 0;
    for(unsigned i = 0; i < (*width) * (*height) * pixel_size; i += pixel_size*4)
    {
        for(unsigned k = 0; k < pixel_size; k++)
        {
            (*newimage)[j + k] = image[i + k];
        }
        j += pixel_size;
        if(i % (*width)*pixel_size == 0)
        {
            i += (*width)*pixel_size;
        }

    }
    return true;
}
