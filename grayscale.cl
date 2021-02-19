__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void grayscale(
          __read_only image2d_t inputImage, 
          __write_only image2d_t outputImage
    ) {
    /* Store each work-item in co-ordinate form 'pos'-->{Row ,Column} */
    const int2 pos = {get_global_id(0), get_global_id(1)};
    uint4 pixel = read_imageui(inputImage, sampler, pos);

    /* unsigned int is equivalent to unsigned char */
    uint4 sum;
    sum = (0.299*pixel.x) + (0.587*pixel.y) + (0.114*pixel.z);
    
    /* Copy the data */ 
    write_imageui(outputImage, pos, sum);
  }

__kernel void filter(
            __read_only image2d_t inputImage,
            __write_only image2d_t outputImage,
            __constant float *filter,
            __private int filterSize
      ) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    const int2 pos = {get_global_id(0), get_global_id(1)};
    int halfWidth = filterSize/2;
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
    int iterFilter = 0;

    /* Iterate over row and column of the image and move the filter pixel by pixel */
    for(int i = -halfWidth; i <= halfWidth; i++){
      pos.y = row +i;
      for(int j = -halfWidth; j <= halfWidth; j++){  
        pos.x = col +j;
        float4 pixel= read_imagef(inputImage, sampler, pos);
        sum.x += pixel.x * filter[iterFilter++];
      }
    }

    /* Copy the data */
    pos.x = col;
    pos.y = row;
    write_imagef(outputImage, pos, sum);
  }

