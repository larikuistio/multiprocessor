__kernel void zncc(__global uchar *left, __global uchar *right, __global uchar *disparity_image, uint w, uint h, int b, int min_d, int max_d)
{
    // Get pixel x and y 
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    int best_disparity = max_d;
    float best_score = -1;

    // Loop through all disparities in range (min_d, max_d) to find the best disparity
    for (int d = min_d; d <= max_d; d++) {

        // Find the average of a window with side length of b around the pixel
        float window_avg_l = 0;
        float window_avg_r = 0;
        for (int i_box = -(b - 1 / 2) ; i_box <= (b - 1 / 2); i_box++) {
            for (int j_box = -(b - 1 / 2) ; j_box <= (b - 1 / 2); j_box++) {
                // If pixel to be checked OOB 
                if (!(i + i_box >= 0) || !(i + i_box < w) || !(j + j_box >= 0) || !(j + j_box < h) || !(i + i_box - d >= 0) || !(i + i_box - d < w)) {
                    continue;
                }
                window_avg_l += left[(j + j_box) * w + (i + i_box)];
                window_avg_r += right[(j + j_box) * w + (i + i_box - d)];
            }
        }
        window_avg_l = native_divide(window_avg_l, b*b);
        window_avg_r = native_divide(window_avg_r, b*b);

        /* Find the ZNCC score of this pixel with formula 
                sum_left_window_deviation*sum_right_window_deviation / 
                (sqrt(sum_left_window_deviation)* sqrt(sum_right_window_deviation))
        */
        float std_l = 0;
        float std_r = 0;
        float score = 0;
        for (int i_box = -(b - 1 / 2) ; i_box <= (b - 1 / 2); i_box++) {
            for (int j_box = -(b - 1 / 2) ; j_box <= (b - 1 / 2); j_box++) {
                // If pixel to be checked OOB 
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
        score = native_divide(score, native_sqrt(std_l) * native_sqrt(std_r));

        // If score with d disparity better than pervious best score for pixel -> update best disparity and score
        if (score > best_score) {
            best_score = score;
            best_disparity = d;
        }
    }
    disparity_image[j * w + i] = (uchar)abs(best_disparity);
}