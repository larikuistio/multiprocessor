#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <limits.h>
#include <math.h>

#define MAX_DISPARITY 64
#define MIN_DISPARITY 0
#define THRESHOLD 2
#define B 9
#define NEIGHBORHOOD_SIZE 256

unsigned char* calcZNCC(unsigned char *left, unsigned char *right, unsigned short width, unsigned short height, unsigned short min_d, unsigned short max_d)
{
    unsigned char* disparity_image = calloc(width*height, sizeof(unsigned char));

    for (size_t j = 0; j < height; j++)
    {
        for (size_t i = 0; i < width; i++)
        {
            unsigned best_disparity = 0;
            double best_zncc = 0.0; 
            for (size_t d = min_d; d < max_d; d++)
            {
                unsigned window_avg_l = 0;
                unsigned window_avg_r = 0;
                for(int y = -(B-1)/2; y < (B-1)/2; y++)
                {
                    for(int x = -(B-1)/2; x < (B-1)/2; x++)
                    {
                        if(!((i + x) < 0 || (i + x) >= width || (j + y) < 0 || (j + y) >= height))
                        {
                            window_avg_l += left[(j + y) * width + (i + x)];
                            window_avg_r += right[(j + y) * width + (i + x)];
                        }
                    }
                }
                window_avg_l /= B*B;
                window_avg_r /= B*B;


                unsigned numerator = 0;
                unsigned denominator_l = 0;
                unsigned denominator_r = 0;
                double zncc_val = 0.0;
                unsigned left_std = 0;
                unsigned right_std = 0;
                for(int y = -(B-1)/2; y < (B-1)/2; y++)
                {
                    for(int x = -(B-1)/2; x < (B-1)/2; x++)
                    {
                        if(!((i + x - d) < 0 || (i + x - d) >= width || (j + y) < 0 || (j + y) >= height))
                        {
                            right_std = (right[(j + y) * width + (i + x - d)] - window_avg_r);
                        }
                        else
                        {
                            right_std = 0;
                        }
                        if(!((i + x) < 0 || (i + x) >= width || (j + y) < 0 || (j + y) >= height))
                        {
                            //printf("%d\n", (j + y) * width + (i + x));
                            left_std = (left[(j + y) * width + (i + x)] - window_avg_l);
                        }
                        else
                        {
                            left_std = 0;
                        }
                        numerator += left_std * right_std;
                        denominator_l += left_std * left_std;
                        denominator_r += right_std * right_std;
                    }
                }
                zncc_val = numerator / (sqrt(denominator_l) * sqrt(denominator_r));

                if(zncc_val > best_zncc)
                {
                    best_disparity = d;
                    best_zncc = zncc_val;
                }
            }
            disparity_image[j * width + i] = best_disparity;
        }
    }
    return disparity_image;
}

unsigned char* crossCheck(unsigned char *disp_map_1, unsigned char *disp_map_2, unsigned short width, unsigned short height, unsigned threshold)
{

	unsigned char* result_map = malloc(width*height);

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

	unsigned char* result = malloc(width*height);

    for (size_t j = 0; j < height; j++) {
        for (size_t i = 0; i < width; i++) {
			result[j * width + i] = disp_map[j * width + i];

            if(disp_map[j * width + i] == 0) {

				// Searching the neighborhood of the pixel
				for (size_t search_area = 1; search_area < (neighborhood_size/2); search_area++) {

					// Sum all the intensities in search area around pixel
					size_t sum = 0;
					for(size_t y = -search_area; y < search_area; y++) {
						for(size_t x = -search_area; x < search_area; x++) {
							// If the pixel we are searching around
							if (y == 0 && x == 0) continue;
							// If searched pixel OOB dont take into account
							if ( (i + x < 0) || (i + x >= width) || (j + y < 0) || (j + y >= height)) 
								continue;
							sum += disp_map[(j+y)*width+(i+x)];
						}
					}
					// If nothing found in search area widen area
					if (sum == 0) continue;

					size_t search_avg = sum / pow((search_area*2+1), 2);
					if (search_avg == 0) search_avg = 1;
					disp_map[j * width + i] = search_avg;
					break;
				}
            }
        }
    }
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
	for (int i = 0; i < width*height; i++) {
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

    unsigned char *disparityLR;
    unsigned char *disparityRL;
    unsigned char *disparityCC;

    decodeImage(inputimg_r, &image_r, &width, &height);
    decodeImage(inputimg_l, &image_l, &width, &height);
    resizeImage(image_r, &rzd_image_r, &width, &height, &resizedWidth, &resizedHeight);
    resizeImage(image_l, &rzd_image_l, &width, &height, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_r, &grayscale_r, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_l, &grayscale_l, &resizedWidth, &resizedHeight);

    disparityLR = calcZNCC(grayscale_l, grayscale_r, resizedWidth, resizedHeight, MIN_DISPARITY, MAX_DISPARITY);
    disparityRL = calcZNCC(grayscale_r, grayscale_l, resizedWidth, resizedHeight, -MAX_DISPARITY, MIN_DISPARITY);

    normalize(disparityLR, resizedWidth, resizedHeight);
    normalize(disparityRL, resizedWidth, resizedHeight);

    disparityCC = crossCheck(disparityLR, disparityRL, resizedWidth, resizedHeight, 8);

    //disparityOcclusion = occlusionFill()

    encodeImage("disparityLR.png", disparityLR, &resizedWidth, &resizedHeight);
    encodeImage("disparityRL.png", disparityRL, &resizedWidth, &resizedHeight);

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