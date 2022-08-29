#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <sutil/vec_math.h>
#include <device_launch_parameters.h>
#include <cuda/util.h>

//Kernel
extern "C" __global__ void postEffect(const float4 * input_image, uchar4 * output_image, unsigned int width, unsigned int height) {
	 int idx_x = 0;
	 int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
	 int idx = 0;

	 if ((idx_x < width) && (idx_y < height)) {
		output_image[idx] =  make_color(input_image[idx]);
	 }
}

__device__ void toneMapping(const float4 color) {

}

