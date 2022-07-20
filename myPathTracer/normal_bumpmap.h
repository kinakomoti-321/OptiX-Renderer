#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
#include <myPathTracer/constant.h>
#include <myPathTracer/random.h>

#include <myPathTracer/BSDF.h>
#include <myPathTracer/RayTrace.h>

static __forceinline__ __device__ float3 normalmap(const float3& n, const float2& uv, const int texID) {
	if (texID == -1) return n;
	float3 t, b;
	tangentSpaceBasis(n, t, b);

	float4 texturecolor = tex2D<float4>(params.textures[texID], uv.x, uv.y);
	float3 shadingnormal = normalize(make_float3(texturecolor.x,texturecolor.y,texturecolor.z ));
	shadingnormal = shadingnormal * 2.0f - 1.0f;
	return normalize(local_to_world(shadingnormal, t, n, b));
}

static __forceinline__ __device__ float3 bumpmap(const float3& n, const float2& uv, const int texID) {
	if (texID == -1) return n;
	float3 t, b;
	tangentSpaceBasis(n, t, b);
	
	float ipsiron = 0.0001f;
	float4 texturecolor = tex2D<float4>(params.textures[texID], uv.x, uv.y);
	float4 texturecolor_du = tex2D<float4>(params.textures[texID], uv.x + ipsiron, uv.y);
	float4 texturecolor_dv = tex2D<float4>(params.textures[texID], uv.x, uv.y + ipsiron);

	float h = texturecolor.x;
	float h_du = texturecolor_du.x;
	float h_dv = texturecolor_dv.x;

	float3 du = normalize(make_float3(1, (h_du - h) / ipsiron, 0));
	float3 dv = normalize(make_float3(0, (h_dv - h) / ipsiron, 1));
	
	float3 shadingnormal = normalize(cross(du, dv));
	return local_to_world(shadingnormal, t, n, b);
}
