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
    
    local uchar leftpix[8][8];
    local uchar rightpix[8][8];

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
    barrier(CLK_LOCAL_MEM_FENCE);

    if(((j+xx) > 0) && ((j+xx) < height) && ((i+yy) > 0) && ((i+yy) < width))
    {
        leftpix[x][y] = left[(int)(j + xx)*width + i + yy];
    }
    else
    {
        leftpix[x][y] = 0;
    }
    if(((j+xx) > 0) && ((j+xx) < height) && ((i+yy-dd) > 0) && ((i+yy-dd) < width))
    {
        rightpix[x][y] = right[(int)(j + xx)*width + i + yy-dd];
    }
    else
    {
        rightpix[x][y] = 0;
    }
    disparities[j*width + i] = max_d;
    zncc_scores[j*width + i] = -1;

    double newl, newr;
    window_avg_l = 0;
    window_avg_r = 0;
    //barrier(CLK_LOCAL_MEM_FENCE);

    barrier(CLK_LOCAL_MEM_FENCE);
    // If pixel to be checked not OOB 
    //if((i + yy >= 0) && (i + yy < width) && (j + xx >= 0) && (j + xx < height) && (i + yy - dd >= 0) && (i + yy - dd < width))
    //{
        atomic_store((volatile __local atomic_double *)window_avg_l, *window_avg_l + leftpix[x][y]);
        atomic_store((volatile __local atomic_double *)window_avg_r, *window_avg_r + rightpix[x][y]);
   //}
    barrier(CLK_LOCAL_MEM_FENCE);
    newl = *window_avg_l / bb;
    newr = *window_avg_r / bb;
    atomic_store((volatile __local atomic_double *)window_avg_l, newl);
    atomic_store((volatile __local atomic_double *)window_avg_r, newr);
    barrier(CLK_LOCAL_MEM_FENCE);
    //barrier(CLK_LOCAL_MEM_FENCE);

    /* Find the ZNCC score of this pixel with formula 
            sum_left_window_deviation*sum_right_window_deviation / 
            (sqrt(sum_left_window_deviation)* sqrt(sum_right_window_deviation))
    */
    
    double zncc_val = 0.0;
    double left_std = 0;
    double right_std = 0;

    left_std = leftpix[x][y] - *window_avg_l;
    right_std = rightpix[x][y] - *window_avg_r;
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_store((volatile __local atomic_double *)denominator_l, *denominator_l + (left_std * left_std));
    atomic_store((volatile __local atomic_double *)denominator_r, *denominator_r + (right_std * right_std));
    atomic_store((volatile __local atomic_double *)zncc_score, *zncc_score + (left_std * right_std));

    //barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);
    zncc_val = native_divide(*zncc_score, native_sqrt(*denominator_l) * native_sqrt(*denominator_r));
    
    // If score with d disparity better than pervious best score for pixel -> update best disparity and score
    
    barrier(CLK_LOCAL_MEM_FENCE);
    while(atomic_flag_test_and_set((volatile __global atomic_flag*)&guard)) {}
    
    if (zncc_val > zncc_scores[j * width + i]) {
        zncc_scores[j*width + i] = zncc_val;
        disparities[j*width + i] = dd;
    }
    atomic_flag_clear((volatile __global atomic_flag*)&guard);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    disparity_image[j * width + i] = (uchar)abs(disparities[j * width + i]);

}