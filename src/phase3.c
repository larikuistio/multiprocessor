#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

#define MAX_DISPARITY 65
#define MIN_DISPARITY 0
#define THRESHOLD 2
#define B 8
#define NEIGHBORHOOD_SIZE 256

unsigned char* calcZNCC(const unsigned char *left, const unsigned char *right, unsigned width, 
    unsigned height, int b, int min_d, int max_d)
{
    unsigned char* disparity_image = (unsigned char*)malloc(width*height);

#pragma omp parallel for shared(left, right, disparity_image) firstprivate(width,height,min_d,max_d, b)
    for (unsigned i = 0; i < width; i++)
    {
        for (unsigned j = 0; j < height; j++)
        {
            int best_disparity = max_d;
            double best_zncc = -1;
            for (int d = min_d; d <= max_d; d++)
            {
                double window_avg_l = 0.0;
                double window_avg_r = 0.0;
                for(int y = -(b - 1 / 2); y <= (b - 1 / 2); y++)
                {
                    for(int x = -(b - 1 / 2); x <= (b - 1 / 2); x++)
                    {
                        if(!(i + y >= 0) || !(i + y < width) || !(j + x >= 0) || !(j + x < height) || !(i + y - d >= 0) || !(i + y - d < width))
                        {
                            continue;
                        }
                        window_avg_l += left[(j + x) * width + (i + y)];
                        window_avg_r += right[(j + x) * width + (i + y - d)];
                    }
                }
                window_avg_l /= b*b;
                window_avg_r /= b*b;
                

                double numerator = 0;
                double denominator_l = 0;
                double denominator_r = 0;
                double zncc_val = 0.0;
                double left_std = 0;
                double right_std = 0;
                for(int y = -(b - 1 / 2); y <= (b - 1 / 2); y++)
                {
                    for(int x = -(b - 1 / 2); x <= (b - 1 / 2); x++)
                    {
                        if(!(i + y >= 0) || !(i + y < width) || !(j + x >= 0) || !(j + x < height) || !(i + y - d >= 0) || !(i + y - d < width))
                        {
                            continue;
                        }
                        left_std = left[(j + x) * width + (i + y)] - window_avg_l;
                        right_std = right[(j + x) * width + (i + y - d)] - window_avg_r;

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

            disparity_image[j * width + i] = (unsigned char)abs(best_disparity);
        }
    }
    return disparity_image;
}



unsigned char* crossCheck(unsigned char *disp_map_1, unsigned char *disp_map_2, 
    unsigned short width, unsigned short height, unsigned threshold)
{

	unsigned char* result_map = malloc(width*height);

#pragma omp parallel for shared(result_map) firstprivate(disp_map_1, disp_map_2, width, height, threshold)
	for (int i = 0; i < width*height; i++) {
		if ((unsigned)abs(disp_map_1[i]-disp_map_2[i]) > threshold) {
			result_map[i] = 0;
		}
		else {
			result_map[i] = disp_map_1[i];
		}
	}
	return result_map;
}

unsigned char* occlusionFill(const unsigned char* disp_map, unsigned width, unsigned height, 
    unsigned neighborhood_size) {

	unsigned char* result = malloc(width*height);

#pragma omp parallel for shared(result) firstprivate(disp_map, width, height, neighborhood_size)
    for (unsigned j = 0; j < height; j++) {
        for (unsigned i = 0; i < width; i++) {
			result[j * width + i] = disp_map[j * width + i];

            if(disp_map[j * width + i] == 0) {
                
				// Searching the neighborhood of the pixel
				for (int search_area = 1; search_area < ((int)neighborhood_size/2); search_area++) {
                    
					// Sum all the intensities in search area around pixel
					int sum = 0;
                    int count = 0;
					for(int y = -search_area; y < search_area; y++) {
						for(int x = -search_area; x < search_area; x++) {
                            
							// If the pixel we are searching around
							if (y == 0 && x == 0) continue;
							// If searched pixel OOB dont take into account
							if ( (i + x < 0) || (i + x >= width) 
                                || (j + y < 0) || (j + y >= height)) 
								continue;

                            unsigned char value = disp_map[(j+y)*width+(i+x)];
                            if (value != 0) count += 1;
							sum += value;

						}
					}
					// If nothing found in search area widen area
					if (sum == 0) continue;
					float search_avg = sum / count;
					if (search_avg < 1) search_avg = 1;
					result[j * width + i] = (int) round(search_avg);
                    
					break;
				}
            }
        }
    }
	return result;
}

void normalize(unsigned char* disparity_img, unsigned width, unsigned height) {

	unsigned max = 0;
	unsigned min = UCHAR_MAX;

#pragma omp parallel for shared(min, max) firstprivate(disparity_img, width, height)
	for (unsigned i = 0; i < width*height; i++) {
		if (disparity_img[i] > max) {
			max = disparity_img[i];
		}
		if (disparity_img[i] < min) {
			min = disparity_img[i];
		}
	}
#pragma omp parallel for firstprivate(disparity_img, min, max, width, height)
	for (unsigned i = 0; i < width*height; i++) {
		disparity_img[i] = 255*(disparity_img[i]- min)/(max-min);
	}
}

int main(int argc, char **argv)
{
    clock_t startprogclk = clock();
	double startprog = queryProfiler();

    if (argc < 2)
    {
        printf("provide input left and input right and output image filenames as arguments\n");
        return EXIT_FAILURE;
    }

    const char *inputimg_l = argv[1];
    const char *inputimg_r = argv[2];

    unsigned char *image_r = 0;
    unsigned char *image_l = 0;
    unsigned char *rzd_image_r = 0;
    unsigned char *rzd_image_l = 0;
    unsigned char *grayscale_r = 0;
    unsigned char *grayscale_l = 0;
    unsigned char *output = 0;

    unsigned width, height, resizedWidth, resizedHeight;

    unsigned char *disparityLR = NULL;
    unsigned char *disparityRL = NULL;
    unsigned char *disparityCC = NULL;
    unsigned char *disparityOF = NULL;

    double start, end;
    clock_t startclk, endclk;

    startclk = clock();
    start = queryProfiler();
    decodeImage(inputimg_r, &image_r, &width, &height);
    decodeImage(inputimg_l, &image_l, &width, &height);
    end = queryProfiler();
    endclk = clock();
    double elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for reading the images: %lf seconds\n", elapsed_time);
    printf("real time used for reading the images: %f seconds\n", end-start);

    startclk = clock();
    start = queryProfiler();
    resizeImage(image_r, &rzd_image_r, &width, &height, &resizedWidth, &resizedHeight);
    resizeImage(image_l, &rzd_image_l, &width, &height, &resizedWidth, &resizedHeight);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for resizing the image: %lf seconds\n", elapsed_time);
    printf("real time used for resizing the images: %f seconds\n", end-start);
    
    startclk = clock();
    start = queryProfiler();
    convertToGrayscale(rzd_image_r, &grayscale_r, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_l, &grayscale_l, &resizedWidth, &resizedHeight);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for converting to grayscale: %lf seconds\n", elapsed_time);
    printf("real time used for converting to grayscale: %f seconds\n", end-start);

    // Start measuring time
    startclk = clock();
    start = queryProfiler();
    disparityLR = calcZNCC((uint8_t*)grayscale_l, (uint8_t*)grayscale_r, resizedWidth, resizedHeight, B, MIN_DISPARITY, MAX_DISPARITY);
    disparityRL = calcZNCC((uint8_t*)grayscale_r, (uint8_t*)grayscale_l, resizedWidth, resizedHeight, B, -MAX_DISPARITY, MIN_DISPARITY);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for calculating zncc: %lf seconds\n", elapsed_time);
    printf("real time used for calculating zncc: %f seconds\n", end-start);

    startclk = clock();
    start = queryProfiler();
    disparityCC = crossCheck(disparityLR, disparityRL, resizedWidth, resizedHeight, THRESHOLD);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for cross checking: %lf seconds\n", elapsed_time);
    printf("real time used for cross checking: %f seconds\n", end-start);
    
    startclk = clock();
    start = queryProfiler();
    disparityOF = occlusionFill(disparityCC, resizedWidth, resizedHeight, NEIGHBORHOOD_SIZE);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for occlusion filling: %lf seconds\n", elapsed_time);
    printf("real time used for occlusion filling: %f seconds\n", end-start);

    start = queryProfiler();
    startclk = clock();
    normalize(disparityLR, resizedWidth, resizedHeight);
    normalize(disparityRL, resizedWidth, resizedHeight);
    normalize(disparityCC, resizedWidth, resizedHeight);
    normalize(disparityOF, resizedWidth, resizedHeight);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for normalizing images: %lf seconds\n", elapsed_time);
    printf("real time used for normalizing images: %f seconds\n", end-start);

    start = queryProfiler();
    startclk = clock();
    lodepng_encode_file("resized_left.png", disparityLR, resizedWidth, resizedHeight, LCT_GREY, 8);
    lodepng_encode_file("resized_right.png", disparityRL, resizedWidth, resizedHeight, LCT_GREY, 8);
    lodepng_encode_file("crosscheck.png", disparityCC, resizedWidth, resizedHeight, LCT_GREY, 8);
    lodepng_encode_file("occlusionfill.png", disparityOF, resizedWidth, resizedHeight, LCT_GREY, 8);
    end = queryProfiler();
    endclk = clock();
    elapsed_time = (endclk-startclk)/(double)CLOCKS_PER_SEC;
    printf("\ncpu time used for writing results to files: %lf seconds\n", elapsed_time); 
    printf("real time used for writing results to files: %f seconds\n", end-start);

    free(image_r);
    free(image_l);
    free(rzd_image_r);
    free(rzd_image_l);
    free(grayscale_r);
    free(grayscale_l);
    free(disparityLR);
    free(disparityRL);
    free(disparityCC);
    free(disparityOF);
    free(output);

    printf("\n\nProgram finished\n");

    clock_t endprogclk = clock();
	double endprog = queryProfiler();

	double elapsed_time_prog = (endprogclk-startprogclk)/(double)CLOCKS_PER_SEC;
	printf("\ncpu time taken by program execution: %lf seconds\n", elapsed_time_prog);
	printf("real time taken by program execution: %f\n", endprog-startprog);

	return EXIT_SUCCESS;
}