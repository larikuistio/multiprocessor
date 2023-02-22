#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

int main(void) {
    const char* filename = "images/im1.png";
    unsigned char* image = 0;
    unsigned width, height;
    decodeImage(filename, &image, &width, &height);


    //const char* filename2 = "images/test1.png";
    const char* filename3 = "images/resized.png";
    unsigned char* image3 = 0;
    unsigned char* image4 = 0;
    unsigned char* image5 = 0;
    unsigned newwidth, newheight;
    resizeImage(image, &image3, &width, &height, &newwidth, &newheight);
    convertToGrayscale(image3, &image4, &newwidth, &newheight, 4);
    const char* filename1 = "images/test.png";
    encodeImage(filename1, image, &width, &height);
    // encodeImage(filename2, image2, &width, &height);
    convertToRGB(image4, &image5, &newwidth, &newheight, 4);
    encodeImage(filename3, image5, &newwidth, &newheight);
    printf("Program finished\n");
    free(image);
    // free(image2);
    free(image3);
    free(image4);
    free(image5);

    return EXIT_SUCCESS;
}
