#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>
#include <myPathTracer/GGX.h>

class Transmit {

public:
	float alpha;
	float ior;

	__device__ float lambda(const float3& v) const {
		float delta = 1 + (alpha * alpha * v.x * v.x + alpha * alpha * v.z * v.z) / (v.y * v.y);
		return (-1.0 + sqrtf(delta)) / 2.0f;
	}

	//Height correlated Smith shadowing-masking
	__device__ float shadowG(const float3& o, const float3& i) {
		return 1.0f / (1.0f + lambda(o) + lambda(i));
	}
	__device__ float shadowG_1(const float3& v) {
		return 1.0f / (1.0f + lambda(v));
	}

	//GGX normal distiribution
	__device__ float GGX_D(const float3& m) {
		float delta = m.x * m.x / (alpha * alpha) + m.z * m.z / (alpha * alpha) + m.y * m.y;
		return 1.0 / (PI * alpha * alpha * delta * delta);
	}

	//Importance Sampling
	//Walter distribution sampling
	__device__ float3 walterSampling(float u, float v) {
		float theta = atanf(alpha * sqrtf(u) / sqrtf(fmaxf(1.0f - u, 0.0f)));
		float phi = 2.0f * PI * v;
		return hemisphereVector(theta, phi);
	}


public:
	__device__ Transmit() {
		ior = 1.5;
		alpha = 0.001f;
	}
	__device__ Transmit(const float ior, const float& in_alpha) :ior(ior) {
		alpha = fmaxf(in_alpha, 0.001f);
	}

	__device__ float3 visibleNormalSampling(const float3& V_, float u, float v) {
		float a_x = alpha, a_y = alpha;
		float3 V = normalize(make_float3(a_x * V_.x, V_.y, a_y * V_.z));

		float3 n = make_float3(0, 1, 0);
		if (V.y > 0.99) n = make_float3(1, 0, 0);
		float3 T1 = normalize(cross(V, n));
		float3 T2 = normalize(cross(T1, V));

		float r = sqrtf(u);
		float a = 1.0f / (1.0f + V.y);
		float phi;
		if (a > v) {
			phi = PI * v / a;
		}
		else {
			phi = PI * (v - a) / (1.0f - a) + PI;
		}

		float P1 = r * cosf(phi);
		float P2 = r * sinf(phi);
		if (a < v) P2 *= V.y;

		float3 N = P1 * T1 + P2 * T2 + sqrtf(fmaxf(1.0f - P1 * P1 - P2 * P2, 0.0f)) * V;

		N = normalize(make_float3(a_x * N.x, N.y, a_y * N.z));
		return N;
	}


	__device__ float3 sampleBSDF(const float3& wo, float3& wi, float& pdf, unsigned int& seed) {
		float3 i = wo;

		//Visible Normal Sampling
		float3 m = visibleNormalSampling(i, rnd(seed), rnd(seed));

		float3 n;
		float ior1, ior2;
		if (i.y > 0) {
			//outside into inside
			ior1 = 1.0f;
			ior2 = ior;
			n = { 0,1,0 };
		}
		else {
			ior1 = ior;
			ior2 = 1.0f;
			n = { 0,-1,0 };
		}

		float _fresnel = fresnel(i, m, ior1, ior2);
		float _fresnel_select = rnd(seed);
		float3 o;
		float3 bsdf;

		if (_fresnel < _fresnel_select) {
			//refraction
			float3 t;
			if (refract(i, m, ior1, ior2, t)) {
				o = t;

				float _D = GGX_D(m);
				float _G = shadowG(i, o);
				float3 _F = make_float3(_fresnel);

				float ni, no;
				ni = ior1;
				no = ior2;

				float im = absDot(i, m);
				float in = absDot(i, n);
				float on = absDot(o, n);
				float om = absDot(o, m);

				float3 half_vector = normalize( - (no * o + ni * i));

				float ih = absDot(i, half_vector);
				float oh = absDot(o, half_vector);

				float half_factor = (ni * dot(i,half_vector) + no * dot(o,half_vector));
				half_factor = half_factor * half_factor;

				float cosine_factor = (ih * oh) / (in * on);

				bsdf = cosine_factor * no * no * (make_float3(1.0) - _F) * _G * _D / (half_factor);
				pdf = _D * shadowG_1(i) * im / (absDot(i, n));
				pdf *= no * no * oh / half_factor;
				pdf *= (1.0 - _fresnel);

				wi = o;

			}
			else {
				o = reflect(-i, m);

				float im = absDot(i, m);
				float in = absDot(i, n);
				float on = absDot(o, n);
				float om = absDot(o, m);

				float _D = GGX_D(m);
				float _G = shadowG(i, o);
				float3 _F = make_float3(_fresnel);

				bsdf = _F * _G * _D / (4.0 * in * on);
				pdf = _D * shadowG_1(i) * im / (absDot(i, n));
				pdf *= 1.0 / (4.0 * im);
				
				wi = o;
			}
		}
		else {
			//reflection
			o = reflect(-i, m);

			float im = absDot(i, m);
			float in = absDot(i, n);
			float on = absDot(o, n);
			float om = absDot(o, m);

			float _D = GGX_D(m);
			float _G = shadowG(i, o);
			float3 _F = make_float3(_fresnel);

			bsdf = _F * _G * _D / (4.0 * in * on);
			pdf = _D * shadowG_1(i) * im / (absDot(i, n));
			pdf *= 1.0 / (4.0 * im);
			pdf *= _fresnel;

			wi = o;
		}

		return bsdf;
	}

	__device__ float pdfBSDF(const float3& wo, const float3& wi) {
		float3 i = wo;
		float3 m = normalize(wi + wo);
		float3 o = wi;
		float3 n = make_float3(0, 1, 0);
		float im = absDot(i, m);
		float D_ = GGX_D(m);
		return D_ * shadowG_1(i) * im / (absDot(i, n) * 4.0f * absDot(m, o));
	}

	//Woが決定した時点でウェイトとしてかかる値
	__device__ float reflect_weight(const float3& wo) {
		return  0.5f;
	}
};
