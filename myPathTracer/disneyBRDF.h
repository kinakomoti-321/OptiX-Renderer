#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/GGX.h>
#include <myPathTracer/Lambert.h>
#include <myPathTracer/BSDF.h>
#include <myPathTracer/color_converter.h>

static __forceinline__ __device__ float f_tSchlick(float wn, float F90) {
	float delta = fmaxf(1.0 - wn, 0.0);
	return 1.0 + (F90 - 1.0) * delta * delta * delta * delta * delta;
}
class DisneyBRDF {
private:
	MaterialParam param;

	float3 rho_sheen;
	float3 rho_specular;
	float3 rho_tint;
	float3 F_s0;
	float aspect;
	float alpha;

	GGX ggx;
	Lambert lan;
	ClearcoatGGX clear;

public:
	__device__ DisneyBRDF() {
		param.diffuse = make_float3(0);
		param.specular = make_float3(0);
		param.metallic = 0;
		param.sheen = 0;
		param.roughness = 0;
		param.ideal_specular = false;
		param.ior = 0;
		rho_sheen = make_float3(0);
		rho_specular = make_float3(0);
		rho_tint = make_float3(0);
		F_s0 = make_float3(0);
		aspect = 0;
	}


	__device__ DisneyBRDF(const MaterialParam& param) :param(param) {
		float sheenTint = 0.0;
		float specularTint = 0.0;

		rho_tint = param.diffuse / RGB_to_Radiance(param.diffuse);
		rho_sheen = lerp(make_float3(1.0), rho_tint, sheenTint);
		rho_specular = lerp(make_float3(1.0), rho_tint, specularTint);

		float specular = 1.0;
		F_s0 = lerp(0.08 * specular * rho_specular, param.diffuse, param.metallic);

		alpha = clamp(param.roughness, 0.01f, 1.0f);

		float clearcoatGloss = 1.0;
		//alpha_clearcoat = lerp(0.1f, 0.001f, clearcoatGloss);

		ggx = GGX(F_s0, alpha);
		lan = Lambert(param.diffuse);
		clear = ClearcoatGGX(make_float3(0.04),param.clearcoat);
	}

	__device__ float3 sampleBRDF(const float3& wi, float3& wo, float& pdf, unsigned int& seed) {

		float3 n = make_float3(0, 1, 0);
		float3 h;
		//Sampling
		{
			float lan_w = lan.reflect_weight(wo) * (1.0 - param.metallic);
			float ggx_w = ggx.reflect_weight(wo);
			float clear_w = clear.reflect_weight(wo) * param.clearcoat;

			float sum_weight = lan_w + ggx_w + clear_w;
			float lan_pdf = lan_w/sum_weight;
			float ggx_pdf = ggx_w / sum_weight;
			float clear_pdf = clear_w / sum_weight;

			float p = rnd(seed);

			float lp, cp, gp;

			if (p < lan_pdf) {
				//Lambert Sampling
				wo = lan.cosineSampling(rnd(seed), rnd(seed), lp);
				lp *= lan_pdf;

				gp = ggx.pdfBSDF(wi, wo);
				gp *= ggx_pdf;

				cp = clear.pdfBSDF(wi, wo);
				cp *= clear_pdf;

				pdf = gp + cp + lp;
			}
			else if(p < lan_pdf + clear_pdf) {
				//Clear Coat Sampling	
				wo = clear.visibleNormalSampling(wi, rnd(seed), rnd(seed));
				wo = reflect(-wi, wo);
				if (wi.y < 0.0) return { 0.0,0.0,0.0 };

				cp = clear.pdfBSDF(wi, wo) * clear_pdf;
				gp = ggx.pdfBSDF(wi, wo) * ggx_pdf;
				lp = lan.pdfBSDF(wi, wo) * lan_pdf;

				pdf = gp + cp + lp;
			}
			else {
				wo = ggx.visibleNormalSampling(wi, rnd(seed), rnd(seed));
				wo = reflect(-wi, wo);
				if (wi.y < 0.0) return { 0.0,0.0,0.0 };

				cp = clear.pdfBSDF(wi, wo) * clear_pdf;
				gp = ggx.pdfBSDF(wi, wo) * ggx_pdf;
				lp = lan.pdfBSDF(wi, wo) * lan_pdf;

				pdf = gp + cp + lp;
			}
		}

		/*
		h = ggx.visibleNormalSampling(wi, rnd(seed), rnd(seed));
		wo = reflect(-wi, h);
		if (wo.y < 0) return { 0,0,0 };
		pdf = ggx.pdfBSDF(wi, wo);
		*/
		/*
		wo = hemisphere_sampling(rnd(seed),rnd(seed),pdf);
		h = normalize(wi + wo);
		*/

		float cosine_d = absDot(wi, h);
		float F_D90 = 0.5 + 2.0 * alpha * cosine_d * cosine_d;
		float F_SS90 = alpha * cosine_d * cosine_d;

		float dot_wi_n = absDot(wi, n);
		float dot_wo_n = absDot(wo, n);
		float dot_wi_m = absDot(wi, h);

		float3 f_diffuse;
		float3 f_subsurface;
		float3 f_sheen;
		float3 f_specular;
		float3 f_clearcoat;

		float f_tsi = f_tSchlick(dot_wi_n, F_D90);
		float f_tso = f_tSchlick(dot_wo_n, F_D90);

		//Diffuse;		
		{
			f_diffuse = param.diffuse * f_tsi * f_tso * invPI;
		}

		//SubSurface
		{
			float deltacos = 1 / (dot_wi_n + dot_wo_n) - 0.5;
			f_subsurface = param.diffuse * invPI * 1.25 * (f_tsi * f_tso * deltacos + 0.5);
		}

		//Specular
		{
			f_specular = ggx.evaluateBSDF(wi, wo);
		}

		//sheen
		{
			float3 wh = normalize(wo + wi);
			float delta = fmaxf(1.0f - absDot(wi, wh), 0.0);
			f_sheen = param.sheen * make_float3(1.0f) * delta * delta * delta * delta * delta;
		}

		//clearcoat
		{
			f_clearcoat = clear.evaluateBSDF(wi,wo);
		}
		return (lerp(f_diffuse,f_subsurface,param.subsurface) + f_sheen)* (1.0f - param.metallic) + f_specular + f_clearcoat;
		//return ggx.sampleBSDF(wi, wo, pdf, seed);
		//return f_sheen;

	}
};
