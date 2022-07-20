#pragma once

#include <optix.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>
#include <myPathTracer/BSDF.h>

extern "C" {
	__constant__ Params params;
}

struct GeoInfo {
	float3 shadingNormal;
	float3 GeoNormal;
	float2 texCooord;
	bool is_light;
	int primID;
};

struct PRD {
	float3 direction; //next Ray Direction
	float3 origin; //next Ray Origin

	float distance;

	unsigned int seed;

	float3 throughput;
	float3 lightColor;

	bool intersection; //Object hit
	bool done; //Light hit

	MaterialParam matparam;
	GeoInfo geoinfo;
};

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1) {
	const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

static __forceinline__ __device__ void* packPointer(void* ptr, unsigned int& i0, unsigned int& i1) {
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
	optixSetPayload_0(float_as_int(p.x));
	optixSetPayload_1(float_as_int(p.y));
	optixSetPayload_2(float_as_int(p.z));
}

static __forceinline__ __device__ PRD* getPRD() {
	const unsigned int u0 = optixGetPayload_0();
	const unsigned int u1 = optixGetPayload_1();
	return reinterpret_cast<PRD*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void TraceOcculution(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin,
	float tmax,
	PRD* prd
) {
	unsigned int u0, u1;
	packPointer(prd, u0, u1);
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		tmin,                // Min intersection distance
		tmax,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		1,                   // SBT offset   -- See SBT discussion
		RAY_TYPE,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		u0,
		u1
	);

}

static __forceinline__ __device__ void RayTrace(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin,
	float tmax,
	PRD* prd
) {
	unsigned int u0, u1;
	packPointer(prd, u0, u1);
	optixTrace(
		params.handle,
		ray_origin,
		ray_direction,
		tmin,                // Min intersection distance
		tmax,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		0,                   // SBT offset   -- See SBT discussion
		RAY_TYPE,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		u0,
		u1
	);

}

