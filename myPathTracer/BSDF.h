#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/GGX.h>
#include <myPathTracer/Lambert.h>
#include <myPathTracer/disneyBRDF.h>

//wo 入射ベクトル
//wi 出射ベクトル


class BSDF {
private:
	MaterialParam param;
	IdealReflect ref;
	GGX ggx;
	Lambert lan;
	DisneyBRDF disney;
public:
	__device__ BSDF(const MaterialParam& param) : param(param) {
		lan = Lambert(param.diffuse);
		float3 spec = lerp(make_float3(0.00), param.diffuse, param.metallic);
		ggx = GGX(spec,param.roughness);
		ref = IdealReflect(param.diffuse);
		disney = DisneyBRDF(param);
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {

		/*
		if (param.metallic < 0.5) {
			return lan.sampleBSDF(wo, wi, pdf, seed);
		}
		else {
			float3 m = ggx.visibleNormalSampling(wo,rnd(seed),rnd(seed));
			wi = reflect(-wo, m);
			if (wi.y < 0) return { 0,0,0 };
			pdf = ggx.pdfBSDF(wo, wi);
			return ggx.evaluateBSDF(wo, wi);
		}
		*/
		return disney.sampleBRDF(wo, wi, pdf, seed);
		/*
		float lan_w = lan.reflect_weight(wo) * (1.0 - param.metallic);
		float ggx_w = ggx.reflect_weight(wo);

		float sum_weight = lan_w + ggx_w;
		float lan_pdf = 0.5;
		float lanbert_pdf;
		float ggx_pdf;

		if (rnd(seed) < lan_pdf) {
			wi = lan.cosineSampling(rnd(seed), rnd(seed), lanbert_pdf);
			lanbert_pdf *= lan_pdf;

			ggx_pdf = ggx.pdfBSDF(wo, wi);
			ggx_pdf *= (1.0 - lan_pdf);

			pdf = lanbert_pdf + ggx_pdf;
		}
		else {
			wi = ggx.visibleNormalSampling(wo, rnd(seed), rnd(seed));
			wi = reflect(-wo, wi);
			if (wi.y < 0.0) {
				pdf = 1.0;
				return { 0.0,0.0,0.0 };
			}
			ggx_pdf = ggx.pdfBSDF(wo, wi);
			ggx_pdf *= (1.0f - lan_pdf);

			lanbert_pdf = lan.pdfBSDF(wo, wi);
			lanbert_pdf *= lan_pdf;

			pdf = lanbert_pdf + ggx_pdf;
		}
		*/
		//return ggx.sampleBSDF(wo, wi, pdf, seed);

		//wi = lan.cosineSampling(rnd(seed), rnd(seed), pdf);
		//wi = ggx.visibleNormalSampling(wo, rnd(seed), rnd(seed));
		//wi = reflect(-wo, wi);
		//ggx_pdf = ggx.pdfBSDF(wo, wi);
		return lan.evaluateBSDF(wo, wi) * (1.0 - param.metallic) + ggx.evaluateBSDF(wo, wi);
		//return lan.evaluateBSDF(wo, wi);
		//wi = ggx.visibleNormalSampling(wo, rnd(seed), rnd(seed));
		//pdf = ggx.pdfBSDF(wo, wi);
		//return ggx.evaluateBSDF(wo, wi);
		//return ggx.sampleBSDF(wo, wi, pdf, seed);
	}

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
		//return lan.evaluateBSDF(wo, wi) * (1.0 - param.metallic) + ggx.evaluateBSDF(wo, wi);
		return disney.evalutateBRDF(wo,wi);
	}

};

