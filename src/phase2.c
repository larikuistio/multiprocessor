#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <limits.h>

#define MAX_DISPARITY 64
#define MIN_DISPARITY 0
#define THRESHOLD 2
#define B 9
#define NEIGHBORHOOD_SIZE 256

unsigned char calcZNCC(unsigned char *left, unsigned char *right, unsigned short width, unsigned short height)
{
    for (size_t j = 0; j < height; j++)
    {
        for (size_t i = 0; i < width; i++)
        {
            for (size_t d = 0; d < MAX_DISPARITY; d++)
            {
                unsigned window_avg_l = 0;
                unsigned window_avg_r = 0;
                for(int y = -(B-1)/2; y < (B-1)/2; y++)
                {
                    for(int x = -(B-1)/2; x < (B-1)/2; x++)
                    {
                        window_avg_l += left[(j + y) * width + (i + x)];
                        window_avg_l += left[(j + y) * width + (i + x)];
                    }
                }
                window_avg_l = B*B;
                window_avg_r = B*B;
            }
        }
    }
}

unsigned char* crossCheck(unsigned char *disp_map_1, unsigned char *disp_map_2, unsigned short width, unsigned short height, unsigned threshold)
{

	unsigned char* result_map = malloc(width*heigth);

	for (int i = 0; i < width*height; i++) {
		if (abs(disp_map_1[i]-disp_map_2[i] > threshold)) {
			result_map[i] = 0;
		}
		else {
			result_map[i] = disp_map_1[i];
		}
	}
	return result_map;
}

int occlusionFill(unsigned char* disp_map, unsigned width, unsigned height, unsigned neighborhood_size) {


}

void normalize(unsigned char* disparity_img, unsigned width, unsigned height) {

	unsigned max = 0;
	unsigned min = UCHAR_MAX;

	for (int i = 0; i < width*height; i++) {
		if (disparity_img[i] > max) {
			max = disparity_img[i];
		}
		if (disparity_img[i] < min) {
			min = disparity_img[i];
		}
	}
	for (i = 0; i < width*height; i++) {
		disparity_img[i] = 255*(disparity_img[i]- min)/(max-min);
	}
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printf("provide input left and input right and output image filenames as arguments\n");
        return EXIT_FAILURE;
    }

    const char *inputimg_r = argv[1];
    const char *inputimg_l = argv[2];
    const char *outputimg = argv[3];

    unsigned char *image_r = 0;
    unsigned char *image_l = 0;
    unsigned char *rzd_image_r = 0;
    unsigned char *rzd_image_l = 0;
    unsigned char *grayscale_r = 0;
    unsigned char *grayscale_l = 0;
    unsigned char *output = 0;

    unsigned width, height, resizedWidth, resizedHeight;

    unsigned char *disparityLR unsigned char *disparityRL

    decodeImage(inputimg_r, &image_r, &width, &height);
    decodeImage(inputimg_l, &image_l, &width, &height);
    resizeImage(image_r, &rzd_image_r, &width, &height, &resizedWidth, &resizedHeight);
    resizeImage(image_l, &rzd_image_l, &width, &height, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_r, &grayscale_r, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_l, &grayscale_l, &resizedWidth, &resizedHeight);

    disparityLR = zncc(grayscale_l, grayscale_r, &resizedWidth, &resizedHeight, MIN_DISPARITY, MAX_DISPARITY);
    disparityRL = zncc(grayscale_r, grayscale_l, &resizedWidth, &resizedHeight, -MAX_DISPARITY, MIN_DISPARITY);

    normalize(disparityLR, resizedWidth, resizedHeight);
    normalize(disparityRL, resizedWidth, resizedHeight);

    // disparityCC = crossChecking(disparityLR, disparityRL, &resizedWidth, &resizedHeight);

    encodeImage("disparityLR.png", disparityLR, &resizedWidth, &resizedHeight)
    encodeImage("disparityRL.png", disparityRL, &resizedWidth, &resizedHeight)

    free(image_r);
    free(image_l);
    free(rzd_image_r);
    free(rzd_image_l);
    free(grayscale_r);
    free(grayscale_l);
    free(disparityLR);
    free(disparityRL);
    free(disparityCC);
    free(output);
}