__kernel void movingfilter5x5(__global uchar* input,
                        __global uchar* output,
                        __private uint width,
                        __private uint height) {
   
   /*uchar filterMatrix[] = {
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0
   };*/
   uchar filterMatrix[] = {
      1, 1, 1, 1, 1,
      1, 3, 3, 3, 1,
      1, 3, 5, 3, 1,
      1, 3, 3, 3, 1,
      1, 1, 1, 1, 1
   };
   ushort filterWidth = 5;
   ushort filterHeight = 5;
   uint imgWidth = width*4;
   uint imgHeight = height*4;
   uint sum = 0;
   uint x = get_global_id(0);
   uint y = 0;

   // Get row by counting how many full rows x has
   while (x > imgWidth) {
      x -= imgWidth;
      y++;
   }

   // If filter box doesnt go OOB
   if (x >= 2 && x <= imgWidth && y >= 2 && y <= imgHeight)
   {
      uint xx = 0;
      uint yy = 0;
      for (uint i = 0; i < filterWidth; i++)
      {
         for (uint j = 0; j < filterHeight; j++)
         {
            xx = x + (i - 2);
            yy = y + (j - 2);
            sum += input[xx + yy * imgWidth] * filterMatrix[i + j * filterWidth];
         }
      }
   }
   else
   {
      uint xx = 0;
      uint yy = 0;
      for (uint i = 0; i < filterWidth; i++)
      {
         for (uint j = 0; j < filterHeight; j++)
         {
            xx = x + (i - 2);
            yy = y + (j - 2);
            if (xx < 0 || xx > imgWidth || yy < 0 || yy > imgHeight)
            {
               sum += 0;
            }
            else
            {
               sum += input[xx + yy * imgWidth] * filterMatrix[i + j * filterWidth];
            }
         }
      }
   }

   output[x + y * imgWidth] = sum/45;
}

