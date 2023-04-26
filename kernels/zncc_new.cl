#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

global atomic_flag guard = ATOMIC_FLAG_INIT;


__kernel void zncc(__global uchar *left, __global uchar *right, 
    __global uchar *disparity_image, uint width, uint height, int b, int min_d, int max_d,
    __global int *disparities, __global double *zncc_scores, __local double *window_avg_l, 
    __local double *window_avg_r, __local double *zncc_score, __local double *denominator_l, 
    __local double *denominator_r)
{
    
    

    // Get pixel x and y 
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    const size_t d = get_global_id(2);
    
    int dd = d + min_d;
    int bn = -(b - 1 / 2);
    int bp = (b - 1 / 2);
    int bb = b*b;
    
    const int y = get_local_id(0);
    const int x = get_local_id(1);
    const int xx = x - bp;
    const int yy = y - bp;

    uchar leftpix = left[(j + xx)*width + i + yy];
    uchar rightpix = right[(j + xx)*width + i + yy - dd];

    disparities[j*width + i] = max_d;
    zncc_scores[j*width + i] = -1;

    double newl, newr;
    barrier(CLK_GLOBAL_MEM_FENCE);


    // If pixel to be checked not OOB 
    if((i + y >= 0) && (i + y < width) && (j + x >= 0) && (j + x < height) && (i + y - dd >= 0) && (i + y - dd < width))
    {
        /*newl = *window_avg_l + leftpix;
        newr = *window_avg_r + rightpix;
        atomic_store((volatile __generic atomic_double *)window_avg_l, newl);
        atomic_store((volatile __generic atomic_double *)window_avg_r, newr);*/

        atomic_store((volatile __local atomic_double *)window_avg_l, *window_avg_l + leftpix);
        atomic_store((volatile __local atomic_double *)window_avg_r, *window_avg_r + rightpix);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    newl = *window_avg_l / bb;
    newr = *window_avg_r / bb;
    atomic_store((volatile __local atomic_double *)window_avg_l, newl);
    atomic_store((volatile __local atomic_double *)window_avg_r, newr);

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Find the ZNCC score of this pixel with formula 
            sum_left_window_deviation*sum_right_window_deviation / 
            (sqrt(sum_left_window_deviation)* sqrt(sum_right_window_deviation))
    */
    
    double zncc_val = 0.0;
    double newdenom_l = 0.0;
    double newdenom_r = 0.0;
    double left_std = 0;
    double right_std = 0;

    left_std = leftpix - *window_avg_l;
    right_std = rightpix - *window_avg_r;
    /*zncc_val = *zncc_score + left_std * right_std;
    newdenom_l = *denominator_l + left_std * left_std;
    newdenom_r = *denominator_r + right_std * right_std;
    atomic_store((volatile __generic atomic_double *)denominator_l, newdenom_l);
    atomic_store((volatile __generic atomic_double *)denominator_r, newdenom_r);
    atomic_store((volatile __generic atomic_double *)zncc_score, zncc_val);*/

    atomic_store((volatile __local atomic_double *)denominator_l, *denominator_l + (left_std * left_std));
    atomic_store((volatile __local atomic_double *)denominator_r, *denominator_r + (right_std * right_std));
    atomic_store((volatile __local atomic_double *)zncc_score, *zncc_score + (left_std * right_std));

    barrier(CLK_LOCAL_MEM_FENCE);

    zncc_val = native_divide(*zncc_score, native_sqrt(*denominator_l) * native_sqrt(*denominator_r));
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // If score with d disparity better than pervious best score for pixel -> update best disparity and score
    atomic_flag_test_and_set((volatile __global atomic_flag*)&guard);
    while (1 == (int)guard) {}
    if (zncc_val > zncc_scores[j * width + i]) {
        atomic_store((volatile __global atomic_double *)&zncc_scores[j * width + i], zncc_val);
        atomic_store((volatile __global atomic_int *)&disparities[j * width + i], dd);
        //zncc_scores[j*width + i] = zncc_val;
        //disparities[j*width + i] = dd;
    }
    atomic_flag_clear((volatile __global atomic_flag*)&guard);
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    disparity_image[j * width + i] = (uchar)abs(disparities[j * width + i]);

}