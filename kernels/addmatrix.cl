__kernel void add_matrix(__global int* a,
                          __global int* b,
                          __global int* results) {
   
   int i = get_global_id(0);
   results[i] = a[i] + b[i];
}

