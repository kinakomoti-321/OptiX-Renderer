#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>

static __forceinline __device__ void gammaConvert(float3& color) {
	color.x = powf(color.x, 1.0f / 2.2f);
	color.y = powf(color.y, 1.0f / 2.2f);
	color.z = powf(color.z, 1.0f / 2.2f);
}

static __forceinline __device__ void gammaInvConvert(float3& color) {
	color.x = powf(color.x, 2.2f);
	color.y = powf(color.y, 2.2f);
	color.z = powf(color.z, 2.2f);
}

static __forceinline __device__ void matrixConvert(float3& color) {
	color.x = powf(color.x, 3.0f / 2.0f);
	color.y = powf(color.y, 4.0f / 5.0f);
	color.z = powf(color.z, 3.0f / 2.0f);
}

static __forceinline__ __device__ float RGB_to_Radiance(const float3& RGB) {
	return 0.2126 * RGB.x + 0.7152 * RGB.y + 0.0722 * RGB.z;
}
