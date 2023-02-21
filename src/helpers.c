#include "helpers.h"

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

bool resizeImage(unsigned char* image, unsigned char* newimage, unsigned* width, unsigned* height, unsigned* newwidth, unsigned* newheight) {
    
    *newwidth = (unsigned)*width/4;
    *newheight = (unsigned)*height/4;
    newimage = malloc((*newwidth)*(*newheight)*4*sizeof(unsigned char));

    if (newimage == NULL) {
        printf("malloc failed\n");
        return false;
    }
    
    
    unsigned char* imgptr = image;
    int j = 0;
    for(unsigned i = 0; i < (*width)*(*height)*4; i += 16)
    {
        newimage[j] = *imgptr;
        imgptr++;
        j++;
        newimage[j] = *imgptr;
        imgptr++;
        j++;
        newimage[j] = *imgptr;
        imgptr++;
        j++;
        newimage[j] = *imgptr;
        imgptr++;
        j++;
        imgptr += 12;
        if(i % 11760 == 0)
        {
            i += 35280;
            imgptr += 35280;
        }
    }
    return true;
}
