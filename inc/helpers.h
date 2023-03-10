#ifndef HELPERS_H
#define HELPERS_H
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "lodepng.h"

void decodeImage(const char* filename, unsigned char** image, unsigned* width, unsigned* height);

void encodeImage(const char* filename, unsigned char* image, unsigned width, unsigned height);


#endif
