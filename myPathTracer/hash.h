#pragma once
#include <cuda_runtime.h>
#include <random.h>

static __host__ __device__ __inline__ float3 hash_int_float3(int i) {
	unsigned int h = i;
	return {rnd(h),rnd(h),rnd(h)};
}
