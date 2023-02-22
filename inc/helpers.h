#ifndef HELPERS_H
#define HELPERS_H
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>

#include "lodepng.h"

void decodeImage(const char* filename, unsigned char** image, unsigned* width, unsigned* height);

void encodeImage(const char* filename, unsigned char* image, unsigned* width, unsigned* height);


bool resizeImage(unsigned char* image, unsigned char** newimage, unsigned* width, unsigned* height, unsigned* newwidth, unsigned* newheight);

#endif
