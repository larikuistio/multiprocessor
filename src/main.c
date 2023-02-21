#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

int main(void) {
    const char* filename = "images/im0.png";
    const char* filename2 = "images/test.png";
    unsigned char* image = 0;
    unsigned width, height;

    decodeImage(filename, image, &width, &height);
    encodeImage(filename2, image, width, height);
    printf("Program finished\n");
    free(image);
    return EXIT_SUCCESS;
}
