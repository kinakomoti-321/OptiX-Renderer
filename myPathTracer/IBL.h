#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/constant.h>

static __device__ float3 IBLRadiance(float3& wi, cudaTextureObject_t& ibl) {
	
	float2 uv;
	float theta = acosf(clamp(wi.y, -1.0f, 1.0f));
	// float phi = std::acos(dir[0] / sqrt(dir[0] * dir[0] + dir[2] * dir[2])) * ((dir[2] < 0.0) ? 1.0f : -1.0f);
	float p = atan2f(wi.z, wi.x);
	float phi = (p < 0) ? (p + PI2) : p;

	uv.y = clamp(theta * invPI, 0.0f, 1.0f);
	uv.x = clamp(phi * invPI2, 0.0f, 1.0f);

	float4 texture_color = tex2D<float4>(ibl, uv.x, uv.y);

	return {texture_color.x,texture_color.y,texture_color.z};
}
