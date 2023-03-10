#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

int main(void) {
    const char* filename = "images/im0.png";
    unsigned char* image = 0;
    unsigned width, height; 
    decodeImage(filename, &image, &width, &height);
    const char* filename1 = "images/test.png";
    encodeImage(filename1, image, &width, &height);
    printf("Program finished\n");
    free(image);
    return EXIT_SUCCESS;
}
