#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"



int calcZNCC() {

}


int crossCheck(unsigned char left, unsigned char image, unsigned short threshold, unsigned short width, unsigned short height) {
  
  for(size_t j = 0; j < height; j++) {
    for(size_t i = 0; i < width; i++) {
      for(size_t d = MA_DISPARI)
    }
  }
}


int occlusionFill(void) {

}

int main(int argc, char **argv) {

    if (argc < 2) {
      printf("provide input and output image files names as arguments\n");
      return EXIT_FAILURE;
    }

    const char* inputimg_r = argv[1];
    const char* inputimg_l = argv[2];
    const char* outputimg = argv[3];

    unsigned char* image_r = 0;
    unsigned char* image_l = 0;
    unsigned char* rzd_image_r = 0;
    unsigned char* rzd_image_l = 0;
    unsigned char* grayscale_r = 0;
    unsigned char* grayscale_l = 0;
    unsigned char* output = 0;

    unsigned width, height, resizedWidth, resizedHeight;

    unsigned charÄdisparityLR 

    decodeImage(inputimg_r, &image_r, &width, &height);
    decodeImage(inputimg_l, &image_l, &width, &height);
    resizeImage(image_r, &rzd_image_r, &width, &height, &resizedWidth, &resizedHeight);
    resizeImage(image_l, &rzd_image_l, &width, &height, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_r, &grayscale_r, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_l, &grayscale_l, &resizedWidth, &resizedHeight);

    disparityLR = 
    disparityRL = 

    free(image)
    free(rzd_image)
    free(grayscale)
    free(output)
}