#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>
#include <myPathTracer/lambert.h>


static __forceinline __device__  float norm2(const float3& v) {
	return length(v) * length(v);
}

static __forceinline __device__  bool refract(const float3& v, const float3& n, float ior1, float ior2,
	float3& r) {
	const float3 t_h = -ior1 / ior2 * (v - dot(v, n) * n);

	// ‘S”½ŽË
	if (norm2(t_h) > 1.0) return false;

	const float3 t_p = -sqrtf(fmaxf(1.0f - norm2(t_h), 0.0f)) * n;
	r = t_h + t_p;

	return true;
}

static __forceinline __device__  float fresnel(const float3& w, const float3& n, float ior1, float ior2) {
	float f0 = (ior1 - ior2) / (ior1 + ior2);
	f0 = f0 * f0;
	float delta = fmaxf(1.0f - dot(w, n), 0.0f);
	return f0 + (1.0f - f0) * delta * delta * delta * delta * delta;
}

class IdealGlass {
private:
	float3 rho;
	float ior;

public:
	__device__ IdealGlass() {
		rho = make_float3(0);
		ior = 1.0;
	}

	__device__ IdealGlass(const float3 & rho, const float& ior) :rho(rho), ior(ior) {}

	__device__ float3 sampleBSDF(const float3 & wo, float3 & wi, float& pdf, unsigned int seed) {
		float ior1, ior2;
		float3 n;
		if (wo.y > 0) {
			ior1 = 1.0f;
			ior2 = ior;
			n = make_float3(0, 1, 0);
		}
		else {
			ior1 = ior;
			ior2 = 1.0f;
			n = make_float3(0, -1, 0);
		}

		const float fr = fresnel(wo, n, ior1, ior2);

		float p = rnd(seed);
		if (p < fr) {
			wi = reflect(-wo, n);
			pdf = fr;
			return fr * rho / fabsf(wi.y);
		}
		else {
			float3 t;
			if (refract(wo, n, ior1, ior2, t)) {
				wi = t;
				pdf = 1;
				return (1.0 - fr) * rho / fabsf(wi.y);
			}
			else {
				wi = reflect(-wo, n);
				pdf = 1;
				return rho / fabsf(wi.y);
			}
		}
	}

	__device__ float3 evalueateBSDF(const float3& wo,const float3& wi) {
		return make_float3(0);
	}

	__device__ float pdfBSDF(const float3& wo,const float3& wi) {
		return 0;
	}

};
