#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#include "helpers.h"

int add_matrices(void) {

    unsigned array_row_size = 100;
    unsigned array_col_size = 100;

    unsigned* arr1 = calloc(array_row_size * array_col_size, sizeof(unsigned));
    unsigned* arr2 = calloc(array_row_size * array_col_size, sizeof(unsigned));
    unsigned* added_arr = calloc(array_row_size * array_col_size, sizeof(unsigned));


    unsigned q;
    for(unsigned i = 0, q = 0 ; i < array_row_size * array_col_size; i++ ) {
        arr1[i] = ++q;
        arr2[i] = ++q;
    }

    addMatrix(arr1, arr2, added_arr, array_row_size, array_col_size);


    for( i = 0; i < array_row_size; i++ ) {
        free(arr1[i]);
        free(arr2[i]);
        free(added_arr[i]);
    }

    free(arr1);
    free(arr2);
    free(added_arr);
    
    return EXIT_SUCCESS;
}

int main(void) {
    // const char* filename = "images/im1.png";
    // unsigned char* image = 0;
    // unsigned width, height;
    // decodeImage(filename, &image, &width, &height);

    // const char* filename1 = "images/test.png";
    // const char* filename2 = "images/test1.png";
    // const char* filename3 = "images/resized.png";
    // unsigned char* image3 = 0;
    // unsigned char* image4 = 0;
    // unsigned char* image5 = 0;
    // unsigned newwidth, newheight;
    // resizeImage(image, &image3, &width, &height, &newwidth, &newheight);
    // convertToGrayscale(image3, &image4, &newwidth, &newheight);
    // convertToRGB(image4, &image5, &newwidth, &newheight);
    // encodeImage(filename1, image, &width, &height);
    // encodeImage(filename2, image2, &width, &height);
    // encodeImage(filename3, image5, &newwidth, &newheight);
    // free(image);
    // free(image2);
    // free(image3);
    // free(image4);
    // free(image5);

    if(add_matrices()) {
        printf("Add matrices failed\n");
        return EXIT_FAILURE;
    
    }
    printf("Program finished\n");
    return EXIT_SUCCESS;
}
