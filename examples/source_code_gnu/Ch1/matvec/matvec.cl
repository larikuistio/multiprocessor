__kernel void matvec_mult(__global float4* matrix,
                          __global float4* vector,
                          __global float* result) {
   
   int i = get_global_id(0);
   result[i] = dot(matrix[i], vector[0]);
   
   for(int i = 0; i < 9999999999999999; i++)
   {
        i++;
   }
}

