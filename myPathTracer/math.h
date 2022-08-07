#pragma once
#include <cuda_runtime.h>
#include <sutil/vec_math.h>

static __forceinline __device__ float3 tangentSpaceBasis(const float3& n, float3& t, float3& b) {
	float3 d = { 0.0f,1.0f,0.0f };
	if (std::abs(n.y) < 0.9f) {
		t = cross(n, d);
	}
	else {
		d = { 0.0f,0.0f,-1.0f };
		t = cross(n, d);
	}

	t = normalize(t);
	b = cross(t, n);
	b = normalize(b);
}

static __forceinline __device__ float3 world_to_local(const float3& v, const float3& lx, const float3& ly, const float3& lz) {
	return {
		v.x * lx.x + v.y * lx.y + v.z * lx.z,
		v.x * ly.x + v.y * ly.y + v.z * ly.z,
		v.x * lz.x + v.y * lz.y + v.z * lz.z,
	};
}

static __forceinline __device__ float3 local_to_world(const float3& v, const float3& lx, const float3& ly, const float3& lz) {
	return {
		v.x * lx.x + v.y * ly.x + v.z * lz.x,
		v.x * lx.y + v.y * ly.y + v.z * lz.y,
		v.x * lx.z + v.y * ly.z + v.z * lz.z,
	};
}

static __forceinline __device__ float3 inverceNormal(const float3& normal, const float3& direction) {
	if (dot(normal, direction) < 0.0) { return normal; }
	return -normal;
}

static __forceinline __device__ float3 hemisphereVector(const float theta, const float phi) {
	return make_float3(cosf(phi) * sinf(theta), cosf(theta), sinf(phi) * sinf(theta));
}

static __forceinline __device__ float absDot(const float3& a,const float3& b) {
	return fmaxf(fabsf(dot(a,b)),0.001);
}
static __forceinline __device__ float lerp(const float& a, const float& b,const float& t) {
	return (1 - t) * a + t * b;
}
namespace BSDFMath {
	static __forceinline__ __device__ float cosTheta(const float3& w) { return w.y; }
	static __forceinline__ __device__ float cos2Theta(const float3& w) { return w.y * w.y; }
	static __forceinline__ __device__ float sinTheta(const float3& w) { return sqrtf(fmaxf(1.0f - cosTheta(w) * cosTheta(w), 0.0f)); }
	static __forceinline__ __device__ float tanTheta(const float3& w) { return sinTheta(w) / cosTheta(w); }
	static __forceinline__ __device__ float tan2Theta(const float3& w) { return tanTheta(w) * tanTheta(w); }

	static __forceinline__ __device__ float cosPhi(const float3& w) {
		float sintheta = sinTheta(w);
		if (sintheta == 0) return 1.0f;
		return clamp(w.x / sintheta, -1.0f, 1.0f);
	}
	static __forceinline__ __device__ float sinPhi(const float3& w) {
		float sintheta = sinTheta(w);
		if (sintheta == 0) return 0.0f;
		return clamp(w.z / sintheta, -1.0f, 1.0f);
	}

	static __forceinline__ __device__ float cosPhi2(const float3& w) { return cosPhi(w) * cosPhi(w); }
	static __forceinline__ __device__ float sinPhi2(const float3& w) { return sinPhi(w) * sinPhi(w); }
}

static __forceinline __device__ float3 AffineConvertPoint(float a[12], float3& point) {
	return make_float3(
		a[0] * point.x + a[1] * point.y + a[2] * point.z + a[3],
		a[4] * point.x + a[5] * point.y + a[6] * point.z + a[7],
		a[8] * point.x + a[9] * point.y + a[10] * point.z + a[11]
	);
}

static __forceinline __device__ float3 AffineConvertVector(float a[12], float3& point) {
	return make_float3(
		a[0] * point.x + a[1] * point.y + a[2] * point.z ,
		a[4] * point.x + a[5] * point.y + a[6] * point.z ,
		a[8] * point.x + a[9] * point.y + a[10] * point.z
	);
}
