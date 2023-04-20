__kernel void crosscheck(__global uchar *disp_map_1, __global uchar *disp_map_2, __global uchar *result_map, int width, int height, int threshold)
{
    const int i = get_global_id(0);

    // If difference between 2 images is higher than threshold set to 0
    if ((unsigned)abs(disp_map_1[i]-disp_map_2[i]) > threshold) {
        result_map[i] = 0;
    }
    // Else set to first image
    else {
        result_map[i] = disp_map_1[i];
    }
}