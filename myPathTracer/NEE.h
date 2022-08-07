#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>
#include <myPathTracer/light_sampling.h>

#include <myPathTracer/BSDF.h>
#include <myPathTracer/RayTrace.h>


static __forceinline__ __device__ float3 NEE(const float3 cameraRayOri, const float3 cameraRayDir, unsigned int seed) {
	float3 ray_origin = cameraRayOri;
	float3 ray_direction = cameraRayDir;
	float3 LTE = { 0.0,0.0,0.0 };
	unsigned int MAX_DEPTH = 100;
	PRD prd;
	prd.seed = seed;
	prd.done = false;
	prd.throughput = { 1.0f,1.0f,1.0f };
	
	float p = 1.0;
	for (int depth = 0; depth < MAX_DEPTH; depth++) {

		//Rossian Roulette
		{
			p = fminf(fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z), 1.0f);
			if (p < rnd(prd.seed)) break;
			prd.throughput /= p;
		}

		//Intersection	
		float3 wo = -ray_direction;
		RayTrace(
			params.handle,
			ray_origin,
			ray_direction,
			0.001f,
			1e16f,
			&prd);

		if (prd.geoinfo.is_light) {
			if (depth == 0) {
				LTE = prd.throughput * prd.lightColor;
			}
			break;
		}
		
		BSDF CurrentBSDF(prd.matparam);
		float3 normal = prd.geoinfo.shadingNormal;
		float3 t, b;
		tangentSpaceBasis(normal, t, b);

		float3 wi;

		{
			PRD light_shot;
			light_shot.done = false;

			float light_pdf;
			float3 light_color;
			float3 light_position = lightPointSampling(prd.seed,light_pdf,light_color);

			float3 light_shadowRay_origin = prd.origin;
			float3 light_shadowRay_direciton = normalize(light_position - light_shadowRay_origin);

			float light_distance = length(light_position - light_shadowRay_origin);
			float ipsiron_distance = 0.001;

			TraceOcculution(
				params.handle,
				light_shadowRay_origin,
				light_shadowRay_direciton,
				0.001f,
				light_distance - ipsiron_distance,
				&light_shot
			);

			if (light_shot.done) {
				float cosine1 = absDot(normal,light_shadowRay_direciton);
				float cosine2 = 1.0f;

				wi = light_shadowRay_direciton;

				float3 local_wo = world_to_local(wo,t,normal,b);
				float3 local_wi = world_to_local(wi,t,normal,b);

				float3 bsdf = CurrentBSDF.evaluateBSDF(local_wo,local_wi);

				float G = cosine2 / (light_distance * light_distance);

				LTE += prd.throughput * (bsdf * G * cosine1 / light_pdf) * light_color;
			}
		}

		//sampling
		float pdf;
		float3 local_wo = world_to_local(wo, t, normal, b);
		float3 local_wi;

		float3 brdf = CurrentBSDF.sampleBSDF(local_wo, local_wi, pdf, prd.seed);
		wi = local_to_world(local_wi, t, normal, b);
		float cosine = absDot(wi, normal);
		
		prd.throughput *= cosine * brdf / pdf;
		ray_origin = prd.origin + prd.geoinfo.GeoNormal * 0.001;
		ray_direction = wi;
	}

	return LTE;
}

