#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>
#include <myPathTracer/lambert.h>

/*
class Phong {
private:
	float3 _specular;
	float _n;

public:
	__device__ Phong() {
		_specular = make_float3(0);
		_n = 0;
	}
	__device__ Phong(float3 specular,float roughness) { 
		_specular = specular;
		_n = 0; 
	}
	
	__device__ float3 phong_sampling(float u, float v,float& pdf) {
		float theta = acosf(powf(1 - u, _n));
		float phi = 2 * PI * u;
		float3 sample_direction = make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
		float pdf_fuctor = (_n + 1) / (2 * PI);
		pdf = pdf_fuctor * powf(sample_direction.y,_n);

		return sample_direction;
	}
	__device__ float3 sampleBSDF(const float3& wo,float3 wi,float& pdf,unsigned int& seed) {
		float3 n = { 0.0,1.0,0.0 };
		wi = reflect(-wo, n);
		pdf = 1.0f;
		return make_float3(1);
		float3 ref = reflect(-wo, make_float3(0, 1, 0));
		float3 ref_t, ref_b;
		tangentSpaceBasis(ref,ref_t, ref_b);

		float3 sample_direction = phong_sampling(rnd(seed),rnd(seed), pdf);

		wi = local_to_world(sample_direction,ref_t, ref ,ref_b);
		//wi = hemisphere_sampling(rnd(seed), rnd(seed), pdf);
		printf("ref %f,%f,%f,wi %f,%f,%f,dot %f \n", ref.x, ref.y, ref.z,wi.x, wi.y, wi.z,dot(ref,wi));

		pdf = abs(wi.y);
		return dot(ref, wi) * make_float3(1);

		if (wi.y < 0) return make_float3(0);
		float3 bsdf = evaluateBSDF(wo, wi);
		if (isnan(bsdf.x) || isnan(bsdf.y) || isnan(bsdf.z) || isnan(pdf)) {
			return make_float3(0);
		}
		//printf("%f \n", pdf);
		//printf("%f,%f,%f \n",bsdf.x, bsdf.y, bsdf.z);
		//return bsdf;
	}

	__device__ float3 evaluateBSDF(const float3&wo ,const float3& wi) {
		float3 ref = reflect(-wo, make_float3(0, 1, 0));
		float specular_power = powf(dot(ref,wi),_n);
		float normalize_fuctor = (_n + 2) / (2 * PI);

		return normalize_fuctor * _specular * specular_power;
	}
};
*/

class Phong {
private:
	float3 rho;
	float _n;
public:
	__device__ Phong() {
		rho = { 0.0,0.0,0.0 };
	}

	__device__ Phong(const float3& rho,float roughness) :rho(rho),_n(roughness) {
		_n = 1000.0 * roughness *roughness;
	}

	__device__ float3 phong_sampling(float u, float v,float& pdf) {
		float theta = acosf(powf(u, 1/(1.0 +_n)));
		float phi = 2.0 * PI * v;
		float3 sample_direction = make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));

		float pdf_fuctor = (_n + 1) / (2 * PI);
		pdf = pdf_fuctor * powf(sample_direction.y,_n);

		return sample_direction;
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {
		float3 n = { 0.0,1.0,0.0 };

		float3 sample_direction = phong_sampling(rnd(seed),rnd(seed), pdf);
		float3 ref = reflect(-wo, n);
		float3 ref_t, ref_b;
		tangentSpaceBasis(ref,ref_t, ref_b);

		wi = local_to_world(sample_direction,ref_t, ref ,ref_b);
		pdf = 1.0;
		if (wi.y < 0)return make_float3(0);
		
		return (_n + 2) / (_n + 1) * rho;
	}

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
		return make_float3(0);
	}
	__device__ float pdfBSDF(const float3& wo,const float3& wi) {
		return 0.0f;
	}
};

class BlinnPhong {
private:
	float3 rho;
	float _n;
public:
	__device__ BlinnPhong() {
		rho = { 0.0,0.0,0.0 };
	}

	__device__ BlinnPhong(const float3& rho,float roughness) :rho(rho),_n(roughness) {
		_n = (2000.0 * roughness *roughness);
	}

	__device__ float3 phong_sampling(float u, float v,float& pdf) {
		float theta = acosf(powf(u, 1/(_n+1)));
		float phi = 2.0 * PI * v;
		float3 sample_direction = make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));

		float pdf_fuctor = (_n + 1) / (2 * PI);
		pdf = pdf_fuctor * powf(sample_direction.y,_n);

		return sample_direction;
	}

	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {
		float3 n = { 0.0,1.0,0.0 };
		
		//half vector sampling
		float3 wh= phong_sampling(rnd(seed),rnd(seed), pdf);

		wi = reflect(-wo,wh);
		float h_jocobian = 4.0 * dot(wh, wi);
		pdf =   (_n + 1)/((2*PI) * h_jocobian) ;

		if (wi.y < 0)return make_float3(0);
		float normalize_factor = (_n + 8) / (8 * PI);
		return make_float3(1) * normalize_factor;
	}

	__device__ float3 evaluateBSDF(const float3& wo,const float3& wi) {
		return make_float3(0);
	}
	__device__ float pdfBSDF(const float3& wo,const float3& wi) {
		return 0.0f;
	}
};

