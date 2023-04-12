#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
// #include <time.h>
#include <omp.h>

#define MAX_DISPARITY 65
#define MIN_DISPARITY 0
#define THRESHOLD 2
#define B 5
#define NEIGHBORHOOD_SIZE 256

uint8_t *calcZNCC(const uint8_t *left, const uint8_t *right, uint32_t w, uint32_t h, int32_t b, int32_t min_d, int32_t max_d)
{

    uint8_t* disparity_image = (uint8_t *) malloc(w*h);


#pragma omp parallel for shared(left, right, disparity_image) firstprivate(w,h,min_d,max_d, b)
{
    for (uint32_t i = 0; i < w; i++) {
        for (uint32_t j = 0; j < h; j++) {

            int32_t best_disparity = max_d;
            float best_score = -1;

            for (int32_t d = min_d; d <= max_d; d++) {
                float window_avg_l = 0;
                float window_avg_r = 0;
                for (int32_t i_box = -(b - 1 / 2) ; i_box <= (b - 1 / 2); i_box++) {
                    for (int32_t j_box = -(b - 1 / 2) ; j_box <= (b - 1 / 2); j_box++) {
                        if (!(i + i_box >= 0) || !(i + i_box < w) || !(j + j_box >= 0) || !(j + j_box < h) || !(i + i_box - d >= 0) || !(i + i_box - d < w))
                        {
                            continue;
                        }
                        window_avg_l += left[(j + j_box) * w + (i + i_box)];
                        window_avg_r += right[(j + j_box) * w + (i + i_box - d)];
                    }
                }
                window_avg_l /= b*b;
                window_avg_r /= b*b;

                float std_l = 0;
                float std_r = 0;
                float score = 0;

                for (int32_t i_box = -(b - 1 / 2) ; i_box <= (b - 1 / 2); i_box++) {
                    for (int32_t j_box = -(b - 1 / 2) ; j_box <= (b - 1 / 2); j_box++) {
                        if (!(i + i_box >= 0) || !(i + i_box < w) || !(j + j_box >= 0) || !(j + j_box < h) || !(i + i_box - d >= 0) || !(i + i_box - d < w))
                        {
                            continue;
                        }

                        float dev_l = left[(j + j_box)* w + (i + i_box)] - window_avg_l;
                        float dev_r = right[(j + j_box) * w + (i + i_box - d)] - window_avg_r;

                        std_l += dev_l * dev_l;
                        std_r += dev_r * dev_r;
                        score += dev_l * dev_r;
                    }
                }

                score /= sqrt(std_l) * sqrt(std_r);

                if (score > best_score) {
                    best_score = score;
                    best_disparity = d;
                }
            }
            disparity_image[j * w + i] = (uint8_t)abs(best_disparity);
        }
    }
}
    return disparity_image;
}

unsigned char* crossCheck(unsigned char *disp_map_1, unsigned char *disp_map_2, 
    unsigned short width, unsigned short height, unsigned threshold)
{

	unsigned char* result_map = malloc(width*height);

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

	for (unsigned i = 0; i < width*height; i++) {
		if (disparity_img[i] > max) {
			max = disparity_img[i];
		}
		if (disparity_img[i] < min) {
			min = disparity_img[i];
		}
	}
	for (unsigned i = 0; i < width*height; i++) {
		disparity_img[i] = 255*(disparity_img[i]- min)/(max-min);
	}
}

double queryProfiler(void)
{
    struct timeval time;
    gettimeofday(&time, 0);
    long seconds = time.tv_sec;
    long microseconds = time.tv_usec;
    double total_time = seconds + microseconds*1e-6;
    return total_time;
}

int main(int argc, char **argv)
{

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

    start = queryProfiler();
    decodeImage(inputimg_r, &image_r, &width, &height);
    decodeImage(inputimg_l, &image_l, &width, &height);
    end = queryProfiler();
    printf("real time used for reading the images: %f\n", end-start);


    start = queryProfiler();
    resizeImage(image_r, &rzd_image_r, &width, &height, &resizedWidth, &resizedHeight);
    resizeImage(image_l, &rzd_image_l, &width, &height, &resizedWidth, &resizedHeight);
    end = queryProfiler();
    printf("real time used for resizing the images: %f\n", end-start);    
    
    start = queryProfiler();
    convertToGrayscale(rzd_image_r, &grayscale_r, &resizedWidth, &resizedHeight);
    convertToGrayscale(rzd_image_l, &grayscale_l, &resizedWidth, &resizedHeight);
    end = queryProfiler();
    printf("real time used for converting to grayscale: %f\n", end-start);

    // Start measuring time
    start = queryProfiler();
    disparityLR = calcZNCC((uint8_t*)grayscale_l, (uint8_t*)grayscale_r, resizedWidth, resizedHeight, B, MIN_DISPARITY, MAX_DISPARITY);
    disparityRL = calcZNCC((uint8_t*)grayscale_r, (uint8_t*)grayscale_l, resizedWidth, resizedHeight, B, -MAX_DISPARITY, MIN_DISPARITY);
    end = queryProfiler();
    printf("real time used for calculating zncc: %f\n", end-start);

    start = queryProfiler();
    disparityCC = crossCheck(disparityLR, disparityRL, resizedWidth, resizedHeight, THRESHOLD);
    end = queryProfiler();
    printf("real time used for cross checking: %f\n", end-start);
    
    start = queryProfiler();
    disparityOF = occlusionFill(disparityCC, resizedWidth, resizedHeight, NEIGHBORHOOD_SIZE);
    end = queryProfiler();
    printf("real time used for occlusion filling: %f\n", end-start);

    start = queryProfiler();
    normalize(disparityLR, resizedWidth, resizedHeight);
    normalize(disparityRL, resizedWidth, resizedHeight);
    normalize(disparityCC, resizedWidth, resizedHeight);
    normalize(disparityOF, resizedWidth, resizedHeight);
    end = queryProfiler();
    printf("real time used for normalizing images: %f\n", end-start);

    start = queryProfiler();
    lodepng_encode_file("resized_left.png", disparityLR, resizedWidth, resizedHeight, LCT_GREY, 8);
    lodepng_encode_file("resized_right.png", disparityRL, resizedWidth, resizedHeight, LCT_GREY, 8);
    lodepng_encode_file("crosscheck.png", disparityCC, resizedWidth, resizedHeight, LCT_GREY, 8);
    lodepng_encode_file("occlusionfill.png", disparityOF, resizedWidth, resizedHeight, LCT_GREY, 8);
    end = queryProfiler();
    printf("real time used for writing results to files: %f\n", end-start);

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
}