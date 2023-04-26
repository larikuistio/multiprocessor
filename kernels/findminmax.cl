__kernel void findminmax(__global uchar *disparity_img, __global int* min, __global int* max)
{
	
    const int i = get_global_id(0);
	
    if (disparity_img[i] > *max) {
		atomic_store((volatile __global atomic_int *)max, disparity_img[i]);
	}
	if (disparity_img[i] < *min) {
		atomic_store((volatile __global atomic_int *)min, disparity_img[i]);
	}

}
