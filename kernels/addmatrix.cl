__kernel void add_matrix(__global int4* a,
                          __global int4* b,
                          __global int4* results) {
   
   int i = get_global_id(0);
   results[i] = a[i] + b[i];
}

