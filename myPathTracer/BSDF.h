#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/GGX.h>
#include <myPathTracer/Lambert.h>

//wo 入射ベクトル
//wi 出射ベクトル

struct MaterialParam {
	float3 diffuse;
	float3 specular;
	float roughness;
	float metallic;
	float sheen;
	float ior;
	bool ideal_specular;
};

class BSDF {
private:
	MaterialParam param;
	Lambert lan;
	GGX ggx;
	IdealReflect ref;
public:
	__device__ BSDF(const MaterialParam& param) : param(param) {
		lan = Lambert(param.diffuse);
		ggx = GGX(param.diffuse, param.roughness);
		ref = IdealReflect(param.diffuse);
	}

	__device__ float3 sampleBSDF(const float3& wo,float3& wi,float& pdf,unsigned int& seed) {
		if (param.metallic < 0.5) {
			return lan.sampleBSDF(wo, wi, pdf, seed);
		}
		else {
			return ggx.sampleBSDF(wo, wi, pdf, seed);
			//return ref.sampleBSDF(wo, wi, pdf, seed);
		}
	}

};

