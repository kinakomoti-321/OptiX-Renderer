#pragma once

#include <optix.h>
#include <cuda/helpers.h>
#include <sutil/vec_math.h>
#include <myPathTracer/RayTrace.h>
#include <myPathTracer/math.h>

static __forceinline__ __device__ float3 lightPointSampling(unsigned int& seed, float& pdf, float3& light_color, float3& light_normal) {
	float p = rnd(seed);

	float selection_pdf = 0;
	unsigned int light_index = 0;
	//Uniform
	/*
	{
		light_index = unsigned int(p * params.light_polyn);
		if (light_index >= params.light_polyn || light_index < 0) light_index = params.light_polyn-1;
		selection_pdf = 1.0f / float(params.light_polyn);

		//printf("light_index %d %f \n",light_index,p);
		
	}
	*/
	//Weighted
	{
		//“ñ•ª’Tõ‚Åoffset‚ð“¾‚é
		int first = 0, len = params.light_polyn;
		while (len > 0) {
			int half = len >> 1, middle = first + half;
			if (params.light_nee_weight[middle] <= p) {
				first = middle + 1;
				len -= half + 1;
			}
			else {
				len = half;
			}
		}
		int offset = first - 1;
		if (offset >= params.light_polyn - 1) {
			light_index = params.light_polyn - 1;
			selection_pdf = 1.0f - params.light_nee_weight[light_index];
		}
		else {
			light_index = offset;
			selection_pdf =  params.light_nee_weight[light_index + 1] - params.light_nee_weight[light_index];
		}
		//printf("light_index %d, light_weight %f offset %d \n",light_index, selection_pdf,offset);
	}

	//printf("sampling \n");
	unsigned int color_id = params.light_colorIndex[light_index];
	unsigned int primitive_id = params.light_faceID[light_index];
	unsigned int instance_id = params.face_instanceID[primitive_id];
	//printf("offset %d, slection_pdf %f \n", primitive_id,selection_pdf);
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

	pdf = selection_pdf / lightArea;
	light_color = params.light_color[color_id];
	return vert;
}


static __forceinline__ __device__ float lightPointPDF(unsigned int primitiveID) {
	unsigned int primitive_id = primitiveID;
	float selection_pdf;
	/*
	//Uniform;
	{
		selection_pdf = 1.0f / float(params.light_polyn);
	}
	*/
	
	//Weighted
	{
		int first = 0, len = params.light_polyn;
		while (len > 0) {
			int half = len >> 1, middle = first + half;
			if (params.light_faceID[middle] <= primitive_id) {
				first = middle + 1;
				len -= half + 1;
			}
			else {
				len = half;
			}
		}
		int offset = first-1;
		unsigned int light_index = offset;
		if (light_index >= params.light_polyn-1) {
			selection_pdf = 1.0 - params.light_nee_weight[light_index];
		}
		else
		{
			selection_pdf =  params.light_nee_weight[light_index + 1] - params.light_nee_weight[light_index];
		}
		//printf("light_index %d, light_weight %f offset %d primID %d \n",light_index, selection_pdf,offset,primitiveID);
	}

	unsigned int instance_id = params.face_instanceID[primitive_id];
	float affine[12];
	getInstanceAffine(affine, instance_id);

	float3 v1 = AffineConvertPoint(affine, params.vertices[primitive_id * 3]);
	float3 v2 = AffineConvertPoint(affine, params.vertices[primitive_id * 3 + 1]);
	float3 v3 = AffineConvertPoint(affine, params.vertices[primitive_id * 3 + 2]);

	float lightArea = length(cross(v2 - v1, v3 - v1)) / 2.0f;

	return  selection_pdf / lightArea;
}

