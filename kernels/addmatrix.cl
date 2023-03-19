__kernel void add_matrix(__global float8* a,
                          __global float8* b,
                          __global float8* results) {
   
   int i = get_global_id(0);
   results[i] = a[i] + b[i];
}

