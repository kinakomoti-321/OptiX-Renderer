#pragma once

#include <optix.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>
#include <myPathTracer/RayTrace.h>
#include <myPathTracer/math.h>

static __forceinline__ __device__ float3 lightPointSampling(unsigned int& seed,float& pdf,float3& light_color,float3& light_normal) {
	float p = rnd(seed);
	unsigned int light_index = unsigned int(p * params.light_polyn);
	if (light_index == params.light_polyn) light_index--;

	unsigned int color_id = params.light_colorIndex[light_index];
	unsigned int primitive_id = params.light_faceID[light_index];
	unsigned int instance_id = params.face_instanceID[primitive_id];
	float affine[12];
	getInstanceAffine(affine, instance_id);

	float3 v1 = AffineConvertPoint(affine, params.vertices[primitive_id * 3]);
	float3 v2 = AffineConvertPoint(affine, params.vertices[primitive_id * 3 + 1]);
	float3 v3 = AffineConvertPoint(affine, params.vertices[primitive_id * 3 + 2]);

	float3 n1 = params.normals[primitive_id * 3];
	float3 n2 = params.normals[primitive_id * 3 + 1];
	float3 n3 = params.normals[primitive_id * 3 + 2];

	float lightArea = length(cross(v2 - v1, v3 - v1)) / 2.0f;

	float u = rnd(seed);
	float v = rnd(seed);

	float f1 = 1.0f - sqrt(u);
	float f2 = sqrt(u) * (1.0f - v);
	float f3 = sqrt(u) * v;

	float3 vert = v1 * f1 + v2 * f2 + v3 * f3;
	float3 normal = normalize(AffineConvertVector(affine,n1 * f1 + n2 * f2 + n3 * f3));
	light_normal = normal;

	pdf = 1.0f / float(params.light_polyn * lightArea);
	light_color = params.light_color[color_id];
	return vert;
}
