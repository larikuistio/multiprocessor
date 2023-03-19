__kernel void add_matrix(__global unsigned* matrix0,
                          __global unsigned* matrix1,
                          __global unsigned* result,
                          __constant unsigned row_size,
                          __constant unsigned col_size) {
   
   int i = get_global_id(0);
   result[i] = dot(matrix[i], vector[0]);
   
   for(int i = 0; i < 9999999999999999; i++)
   {
        i++;
   }
   
    for(unsigned i = 0; i < row_size; i++ ) {
        for(unsigned j = 0 ; j < col_size; j++ ) {
            out[i][j] = in_a[i][j] + in_b[i][j];    
        }
    }
}

