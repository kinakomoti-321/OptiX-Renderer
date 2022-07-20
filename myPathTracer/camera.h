#pragma once

#include <cuda_runtime.h>
#include <sutil/vec_math.h>
#include <myPathTracer/math.h>
class Camera{
private:
    float3 origin;
    float3 cameraDir;
    float3 cameraUp;
    float3 cameraSide;
    float p;
public:
    __device__ Camera(const float3& origin, const float3& cameraDir, const float p) : origin(origin), cameraDir(cameraDir), p(p) {
        float3 t, b;
        tangentSpaceBasis(cameraDir, t, b);
        cameraUp = t;
        cameraSide = b;
    }

    __device__ void getCameraRay(const float2& uv, float3& rayOri,float3& rayDir, float& weight){
        rayOri = origin;
        rayDir = normalize(cameraDir * p + cameraSide * uv.y + cameraUp * uv.x);
        weight = 1.0;
    }
};
/*
static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction, unsigned int& seed)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	const float2 d = 2.0f * make_float2(
		static_cast<float>(idx.x + rnd(seed) - 0.5f) / static_cast<float>(dim.x),
		static_cast<float>(idx.y + rnd(seed) - 0.5f) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	const float3 U = params.cam_u;
	const float3 V = params.cam_v;
	const float3 W = params.cam_w;
	const float2 d = 2.0f * make_float2(
		static_cast<float>(idx.x) / static_cast<float>(dim.x),
		static_cast<float>(idx.y) / static_cast<float>(dim.y)
	) - 1.0f;

	origin = params.cam_eye;
	direction = normalize(d.x * U + d.y * V + W);
}
*/
