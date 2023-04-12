__kernel void occlusionfill(__global uchar *disp_map, __global uchar *result, int width, int height, int neighborhood_size)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

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