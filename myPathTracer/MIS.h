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

static __forceinline__ __device__ float3 MIS(const float3 cameraRayOri, const float3 cameraRayDir,
	unsigned int seed,float3& albedo) {

	float3 ray_origin = cameraRayOri;
	float3 ray_direction = cameraRayDir;
	float3 LTE = { 0.0,0.0,0.0 };
	unsigned int MAX_DEPTH = 100;
	PRD prd;
	prd.seed = seed;
	prd.done = false;
	prd.throughput = { 1.0f,1.0f,1.0f };

	bool albedo_frag = false;

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

			albedo = (albedo_frag) ? albedo : prd.lightColor;
			break;
		}

		BSDF CurrentBSDF(prd.matparam);
		if (!albedo_frag && !(prd.matparam.roughness < 0.1 || prd.matparam.ideal_specular)) {
			albedo = prd.matparam.diffuse;
			albedo_frag = true;
		}

		float3 normal = prd.geoinfo.shadingNormal;
		float3 t, b;
		tangentSpaceBasis(normal, t, b);

		float3 wi;

		{
			PRD light_shot;
			light_shot.intersection = false;

			float light_pdf;
			float3 light_color;
			float3 light_normal;
			bool is_direction = false;
			float3 light_position = lightPointSampling(prd.seed,light_pdf,light_color,light_normal,is_direction);

			float3 light_shadowRay_origin = prd.origin;
			//float3 light_shadowRay_direciton =  normalize(light_position - light_shadowRay_origin);
			float3 light_shadowRay_direciton = (is_direction) ? - light_normal : normalize(light_position - light_shadowRay_origin);
			//printf("(%f,%f,%f,%d)\n", light_normal.x, light_normal.y, light_normal.z,is_direction);

			float light_distance = (is_direction) ? 1e16f :length(light_position - light_shadowRay_origin);
			float ipsiron_distance = 0.001;

			TraceOcculution(
				params.handle,
				light_shadowRay_origin,
				light_shadowRay_direciton,
				0.001f,
				light_distance - ipsiron_distance,
				&light_shot
			);

			if (!light_shot.intersection) {
				float cosine1 = absDot(normal, light_shadowRay_direciton);
				float cosine2 = absDot(light_normal, -light_shadowRay_direciton);

				wi = light_shadowRay_direciton;

				float3 local_wo = world_to_local(wo, t, normal, b);
				float3 local_wi = world_to_local(wi, t, normal, b);

				float3 bsdf = CurrentBSDF.evaluateBSDF(local_wo, local_wi);

				float G = (is_direction) ? 1.0f : cosine2 / (light_distance * light_distance);
				
				float pt_pdf = (is_direction)? 0.0f : G * CurrentBSDF.pdfBSDF(local_wo,local_wi);

				float miswight = light_pdf / (light_pdf + pt_pdf);
				//miswight = 1;
				//printf("mis_weight(%f) %d %f\n", miswight,is_direction,light_pdf);
				LTE += prd.throughput * miswight * (bsdf * G * cosine1 / light_pdf) * light_color;
			}
		}

		//PT
		{
			float pt_pdf;
			float3 local_wo = world_to_local(wo, t, normal, b);
			float3 local_wi;

			float3 brdf = CurrentBSDF.sampleBSDF(local_wo, local_wi,pt_pdf, prd.seed);
			wi = local_to_world(local_wi, t, normal, b);
			float cosine1 = absDot(wi, normal);
			float3 pt_ray_origin = prd.origin + wi* 0.001;
			float3 pt_ray_direction = wi;

			PRD pt_light_hit;
			pt_light_hit.intersection = false;

			RayTrace(
				params.handle,
				pt_ray_origin,
				pt_ray_direction,
				0.001f,
				1e16f,
				&pt_light_hit);

			if (pt_light_hit.intersection) {
				if (pt_light_hit.geoinfo.is_light) {
					float cosine2 = absDot(-wi, pt_light_hit.geoinfo.shadingNormal);
					float light_distance = pt_light_hit.distance;

					float invG = light_distance * light_distance / cosine2;

					float lightPdf = (prd.matparam.ideal_specular) ? 0 : lightPointPDF(pt_light_hit.geoinfo.primID) * invG;
					float mis_weight =(prd.matparam.ideal_specular) ? 1 : pt_pdf / (pt_pdf + lightPdf);
					
					LTE += prd.throughput * mis_weight * cosine1 * pt_light_hit.lightColor * brdf / pt_pdf;
					//LTE += prd.throughput * cosine1 * pt_light_hit.lightColor * brdf / pt_pdf;
				}
			}
			else {
				LTE += prd.throughput * brdf * cosine1 *  pt_light_hit.lightColor / pt_pdf;
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
		ray_origin = prd.origin + wi * 0.001;
		ray_direction = wi;
	}

	return LTE;

}
