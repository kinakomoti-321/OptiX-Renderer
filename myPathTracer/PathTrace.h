#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>

#include <myPathTracer/BSDF.h>
#include <myPathTracer/RayTrace.h>



static __forceinline__ __device__ float3 PathTrace(const float3 cameraRayOri, const float3 cameraRayDir, unsigned int seed) {
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
	
		p = fminf(fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z), 1.0f);
		if (p < rnd(prd.seed)) break;
		prd.throughput /= p;

		float3 wo = -ray_direction;
		RayTrace(
			params.handle,
			ray_origin,
			ray_direction,
			0.001f,
			1e16f,
			&prd);

		if (prd.geoinfo.is_light) {
			LTE = prd.throughput * prd.lightColor;
			break;
		}
		
		BSDF CurrentBSDF(prd.matparam);
		float3 normal = prd.geoinfo.shadingNormal;
		float3 t, b;
		tangentSpaceBasis(normal, t, b);

		float3 wi;

		//sampling
		float pdf;
		float u1 = rnd(prd.seed);
		float u2 = rnd(prd.seed);
		float3 local_wo = world_to_local(wo, t, normal, b);
		float3 local_wi;

		float3 brdf = CurrentBSDF.sampleBSDF(local_wo, local_wi, pdf, prd.seed);
		wi = local_to_world(local_wi, t, normal, b);
		float cosine = absDot(wi, normal);
		
		prd.throughput *= cosine * brdf / pdf;
		//prd.throughput = prd.matparam.diffuse;
		//LTE = prd.geoinfo.shadingNormal;
		ray_origin = prd.origin + wi * 0.0001;
		ray_direction = wi;
	}

	return LTE;
}

