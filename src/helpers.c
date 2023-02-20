#include "lodepng.h"


unsigned char* decodeImage(const char* filename, unsigned char* image, unsigned* width, unsigned* height) {

	unsigned error;
	unsigned char* png = 0;
	size_t pngsize;

	error = lodepng_load_file(&png, &pngsize, filename);
	if(!error) error = lodepng_decode32(&image, &width, &height, png, pngsize);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	free(png);

	return image;
}