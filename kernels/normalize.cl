__kernel void normalize(__global uchar *disparity_img)
{
    const int i = get_global_id(0);

    if ((unsigned)abs(disp_map_1[i]-disp_map_2[i]) > threshold) {
        result_map[i] = 0;
    }
    else {
        result_map[i] = disp_map_1[i];
    }
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