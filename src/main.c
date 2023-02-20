#include "helpers.h"

void main(void) {
    const char* filename = "images/im0.png";
    unsigned char* image = 0;
    unsigned width, height; 
    decodeImage(filename, image, &width, &height);
    
    printf("Program finished\n");
    free(image);
}
