#include <cuda/helpers.h>
#include <cooperative_groups.h>
#include <sutil/vec_math.h>

//Kernel
/*
__global__ void postEffect(const float4* input_image, uchar4* output_image,unsigned int width, unsigned int height) {
	 int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
	 int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	 int idx = 0;

	 if ((idx_x < width) && (idx_y < height)) {
		output_image[idx] =  make_color(input_image[idx]);
	 }

}

__device__ void toneMapping(const float4 color) {

}

*/
