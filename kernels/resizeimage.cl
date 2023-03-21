__kernel void resizeimage(__global uchar* input,
                        __global uchar* output) {
   
    uint i = get_global_id(0);
    uint j = get_global_id(1);
    if (j % 4 == 0) {
        if (0 == i % 16)
        {
            output[(j*2940)/4 + i/4] = input[j*2940*4 + i];
            output[(j*2940)/4 + i/4 + 1] = input[j*2940*4 + i+1];
            output[(j*2940)/4 + i/4 + 2] = input[j*2940*4 + i+2];
            output[(j*2940)/4 + i/4 + 3] = input[j*2940*4 + i+3];
        }
    }
}

