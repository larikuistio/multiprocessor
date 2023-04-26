__kernel void zncc(__global uchar *left, __global uchar *right, __global uchar *disparity_image, 
    uint width, uint height, int b, int min_d, int max_d)
{
    // Get pixel x and y 
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    int best_disparity = max_d;
    double best_zncc = -1;

    // Loop through all disparities in range (min_d, max_d) to find the best disparity
    for (int d = min_d; d <= max_d; d++) {

        // Find the average of a window with side length of b around the pixel
        double window_avg_l = 0;
        double window_avg_r = 0;
        for (int y = -(b - 1 / 2) ; y <= (b - 1 / 2); y++) {
            for (int x = -(b - 1 / 2) ; x <= (b - 1 / 2); x++) {
                // If pixel to be checked OOB 
                if(!(i + y >= 0) || !(i + y < width) || !(j + x >= 0) || !(j + x < height) || !(i + y - d >= 0) || !(i + y - d < width))
                {
                    continue;
                }
                window_avg_l += left[(j + x) * width + (i + y)];
                window_avg_r += right[(j + x) * width + (i + y - d)];
            }
        }
        window_avg_l = native_divide(window_avg_l, b*b);
        window_avg_r = native_divide(window_avg_r, b*b);

        /* Find the ZNCC score of this pixel with formula 
                sum_left_window_deviation*sum_right_window_deviation / 
                (sqrt(sum_left_window_deviation)* sqrt(sum_right_window_deviation))
        */
        
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
        zncc_val = native_divide(numerator, native_sqrt(denominator_l) * native_sqrt(denominator_r));

        // If score with d disparity better than pervious best score for pixel -> update best disparity and score
        if (zncc_val > best_zncc) {
            best_disparity = d;
            best_zncc = zncc_val;
        }
    }
    disparity_image[j * width + i] = (uchar)abs(best_disparity);
}