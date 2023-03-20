__kernel void grayscale(__global uint* input,
                        __global uint* output) {
   
    uint i = get_global_id(0);
    if (i % 4 == 0)
    {
        output[i] = 0.2126*input[i] + 0.7152*input[i+1] + 0.0722*input[i+2];
    }
}

