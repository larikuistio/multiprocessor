__kernel void grayscale(__global uchar* input,
                        __global uchar* output) {
   
    uint i = get_global_id(0);

    // Input image is RGBA so take only every 4 bytes 
    if (i % 4 == 0)
    {
        output[i/4] = 0.2126*input[i] + 0.7152*input[i+1] + 0.0722*input[i+2];
        /*output[i] = 0.2126*input[i] + 0.7152*input[i+1] + 0.0722*input[i+2];
        output[i+1] = 0.2126*input[i] + 0.7152*input[i+1] + 0.0722*input[i+2];
        output[i+2] = 0.2126*input[i] + 0.7152*input[i+1] + 0.0722*input[i+2];
        output[i+3] = 255;*/
    }
}

