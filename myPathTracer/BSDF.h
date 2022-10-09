#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/GGX.h>
#include <myPathTracer/Lambert.h>
#include <myPathTracer/disneyBRDF.h>
#include <myPathTracer/specular.h>
#include <myPathTracer/Phong.h>

//wo 入射ベクトル
//wi 出射ベクトル


class BSDF {
private:
	MaterialParam param;
	IdealReflect ref;
	IdealGlass glass;
	GGX ggx;
	Lambert lan;
	DisneyBRDF disney;
	Phong phong;
	BlinnPhong blinnphong;

public:
	__device__ BSDF(const MaterialParam& param) : param(param) {
		lan = Lambert(param.diffuse);
		float3 spec = lerp(make_float3(0.00), param.diffuse, param.metallic);
		ggx = GGX(spec,param.roughness);
		ref = IdealReflect(param.diffuse);
		glass = IdealGlass(param.diffuse,param.ior);
		disney = DisneyBRDF(param);
		phong = Phong(param.diffuse, param.roughness);
		blinnphong = BlinnPhong(param.diffuse, param.roughness);
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {

		return blinnphong.sampleBSDF(wo,wi,pdf,seed);
		/*
		if (param.ideal_specular) {
			return glass.sampleBSDF(wo, wi, pdf, seed);
		}
		//return ggx.sampleBSDF(wo, wi, pdf, seed);
		return disney.sampleBRDF(wo, wi, pdf, seed);
		*/
	}

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {

		if (param.ideal_specular) {
			return glass.evalueateBSDF(wo, wi);
		}

		return disney.evalutateBRDF(wo,wi);
	}
	
	__device__ float pdfBSDF(const float3& wo, const float3& wi) {

		if (param.ideal_specular) {
			return glass.pdfBSDF(wo, wi);
		}

		return disney.pdfBSDF(wo, wi);
	}
};

