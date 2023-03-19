#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#include "helpers.h"

int main(void) {
    const char* img0 = "images/im0.png";
    const char* img1 = "images/im1.png";
    unsigned char* image0 = 0;
    unsigned char* image1 = 0;
    unsigned width0, height0;
    unsigned width1, height1;
    decodeImage(img0, &image0, &width0, &height0);
    decodeImage(img1, &image1, &width1, &height1);

    const char* resized0 = "images/resized0.png";
    const char* resized1 = "images/resized1.png";
    const char* grayscale0 = "images/grayscale0.png";
    const char* grayscale1 = "images/grayscale1.png";
    unsigned char* rsized0 = 0;
    unsigned char* rsized1 = 0;
    unsigned char* gscale0 = 0;
    unsigned char* gscale1 = 0;
    unsigned char* gscalergb0 = 0;
    unsigned char* gscalergb1 = 0;
    unsigned newwidth0, newheight0;
    unsigned newwidth1, newheight1;
    resizeImage(image0, &rsized0, &width0, &height0, &newwidth0, &newheight0);
    resizeImage(image1, &rsized1, &width1, &height1, &newwidth1, &newheight1);
    convertToGrayscale(rsized0, &gscale0, &newwidth0, &newheight0);
    convertToGrayscale(rsized1, &gscale1, &newwidth1, &newheight1);
    convertToRGB(gscale0, &gscalergb0, &newwidth0, &newheight0);
    convertToRGB(gscale1, &gscalergb1, &newwidth1, &newheight1);
    encodeImage(resized0, rsized0, &newwidth0, &newheight0);
    encodeImage(resized1, rsized1, &newwidth1, &newheight1);
    encodeImage(grayscale0, gscalergb0, &newwidth0, &newheight0);
    encodeImage(grayscale1, gscalergb1, &newwidth1, &newheight1);
    free(image0);
    free(image1);
    free(rsized0);
    free(rsized1);
    free(gscale0);
    free(gscale1);
    free(gscalergb0);
    free(gscalergb1);

    printf("Program finished\n");
    return EXIT_SUCCESS;
}
