__kernel void normalize(__global uchar *input, __global uchar *output, __global int *min, __global int *max)
{
    const int i = get_global_id(0);
    output[i] = 255*(input[i]- (*min))/((*max)-(*min));
}
