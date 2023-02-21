#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>


int main(void) {
    const char* filename = "images/im0.png";
    unsigned char* image = 0;
    unsigned width, height;
    decodeImage(filename, &image, &width, &height);

    unsigned char* image2 = malloc(23708160 * sizeof(unsigned char));
    if (image2 == NULL) {
        printf("malloc failed\n");
        return EXIT_FAILURE;
    }
    unsigned char* image1 = image;
    for(int i = 0; i < 23708160; i++)
    {
        image2[i] = *image1;
        image1++;
    }

    const char* filename1 = "images/test.png";
    const char* filename2 = "images/test1.png";
    const char* filename3 = "images/resized.png";
    unsigned char* image3 = 0;
    unsigned newwidth, newheight;
    resizeImage(image, image3, &width, &height, &newwidth, &newheight);
    encodeImage(filename1, image, &width, &height);
    encodeImage(filename2, image2, &width, &height);
    encodeImage(filename3, image3, &newwidth, &newheight);
    printf("Program finished\n");
    free(image);
    free(image1);
    free(image2);
    free(image3);
    return EXIT_SUCCESS;
}
