#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <limits.h>
#include <math.h>

#define MAX_DISPARITY 64
#define MIN_DISPARITY 0
#define THRESHOLD 2
#define B 9

unsigned char calcZNCC(unsigned char *left, unsigned char *right, unsigned short width, unsigned short height, unsigned short min_d, unsigned short max_d)
{
    for (size_t j = 0; j < height; j++)
    {
        for (size_t i = 0; i < width; i++)
        {
            for (size_t d = min_d; d < max_d; d++)
            {
                unsigned int window_avg_l = 0;
                unsigned int window_avg_r = 0;
                for(int y = -(B-1)/2; y < (B-1)/2; y++)
                {
                    for(int x = -(B-1)/2; x < (B-1)/2; x++)
                    {
                        if((i + x) < 0 || (i + x) >= width || (j + y) < 0 || (j + y) >= height)
                        {
                            continue;
                        }
                        else
                        {
                            window_avg_l += left[(j + y) * width + (i + x)];
                            window_avg_r += right[(j + y) * width + (i + x - d)];
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
                        if((i + x) < 0 || (i + x) >= width || (j + y) < 0 || (j + y) >= height)
                        {
                            continue;
                        }
                        else
                        {
                            left_std = (left[(j + y) * width + (i + x)] - window_avg_l);
                            right_std = (right[(j + y) * width + (i + x - d)] - window_avg_r);
                            numerator += left_std * right_std;
                            denominator_l += left_std * left_std;
                            denominator_r += right_std * right_std;
                        }
                    }
                }
                zncc_val = numerator / (sqrt(denominator_l) * sqrt(denominator_r));
            }


        }
    }
}

int crossCheck(unsigned char *left, unsigned char *image, unsigned short width, unsigned short height)
{

    for (size_t j = 0; j < height; j++)
    {
        for (size_t i = 0; i < width; i++)
        {
            for (size_t d = 0; d < MAX_DISPARITY; d++)
            {
                unsigned int sum = 0;
                for(int y = B/2; y < B/2; y++)
                {
                    for(int x = B/2; x < B/2; x++)
                    {
                        sum += 
                    }
                }
            }
        }
    }
}

int occlusionFill(void)
{
}

int normalize(unsigned char* disparity_img, unsigned width, unsigned height) {

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
		disparity_img[i] = 255*(disparity_img[i]- min)/(max-min)
	}
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        printf("provide input and output image files names as arguments\n");
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

    disparityLR = zncc(grayscale_r, grayscale_l, &resizedWidth, &resizedHeight, 9);
    disparityRL = zncc(grayscale_l, grayscale_r, &resizedWidth, &resizedHeight, 9);

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