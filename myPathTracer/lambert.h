#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>

static __forceinline__ __device__ float3 hemisphere_sampling(float u1, float u2, float& pdf) {
	const float theta = acosf(fmaxf(1.0f - u1, 0.0f));
	const float phi = 2.0f * PI * u2;
	pdf = 1.0f / (2.0f * PI);
	return make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
}

class Lambert {
private:
	float3 rho;


public:
	__device__ Lambert(){
		rho = { 0.0,0.0,0.0 };
	}
	__device__ Lambert(const float3& rho) :rho(rho) {}

	__device__ float3 sampleBSDF(const float3& wo,float3& wi,float& pdf,unsigned int& seed) {
		wi = cosineSampling(rnd(seed), rnd(seed), pdf);
		return rho * invPI;
	}
	
	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
		return rho * invPI;
	}

	__device__ float pdfBSDF(const float3& wo,const float3& wi) {
		return fabsf(wo.y);
	}

    __device__ float reflect_weight(const float3& wo) {
        return  0.5f;
    }

	__device__ float3 cosineSampling(float u, float v, float& pdf) {
		const float theta =
			acosf(1.0f - 2.0f * u) / 2.0f;
		const float phi = 2.0f * PI * v;
		pdf = cosf(theta) / PI;
		return make_float3(cosf(phi) * sinf(theta), cosf(theta),
			sinf(phi) * sinf(theta));
	}
	
};
