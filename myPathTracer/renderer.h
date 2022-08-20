#pragma once

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#include <myPathTracer/myPathTracer.h>
#include <myPathTracer/debugLog.h>
#include <myPathTracer/material.h>
#include <myPathTracer/texture.h>
#include <myPathTracer/animation.h>
#include <myPathTracer/Denoiser.h>
#include <myPathTracer/PostEffect.h>

struct CameraStatus {
	float3 origin;
	float3 direciton;
	float f;
	int cameraAnimationIndex;
};

struct FrameData {
	float maxFrame;
	float minFrame;
	int fps;
};
template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

/*
inline float3 toSRGB(const float3& col) {
	float  invGamma = 1.0f / 2.4f;
	float3 powed = make_float3(std::pow(col.x, invGamma), std::pow(col.y, invGamma), std::pow(col.z, invGamma));
	return make_float3(
		col.x < 0.0031308f ? 12.92f * col.x : 1.055f * powed.x - 0.055f,
		col.y < 0.0031308f ? 12.92f * col.y : 1.055f * powed.y - 0.055f,
		col.z < 0.0031308f ? 12.92f * col.z : 1.055f * powed.z - 0.055f);
}
*/

inline unsigned char quantizeUnsignedChar(float x) {
	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)std::min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

void float4ConvertColor(float4* data, uchar4* color, unsigned int width, unsigned int height) {
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			unsigned int idx = i + width * j;
			float3 col = make_float3(data[idx]);
			col = toSRGB(col);

			color[idx] = make_uchar4(
				quantizeUnsignedChar(col.x),
				quantizeUnsignedChar(col.y),
				quantizeUnsignedChar(col.z),
				(unsigned char)255);
		}
	}
}
void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
	cam.setEye({ 5.0f, 0.0f, 0.0f });
	cam.setLookat({ 0.0f, 0.0f, 0.0f });
	cam.setUp({ 0.0f, 1.0f, 0.0f });
	cam.setFovY(45.0f);
	cam.setAspectRatio((float)width / (float)height);
}


void printUsageAndExit(const char* argv0)
{
	std::cerr << "Usage  : " << argv0 << " [options]\n";
	std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
	std::cerr << "         --help | -h                 Print this usage message\n";
	std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
	exit(1);
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}


struct RendererState {
	//Frame�Ȃǂ̊Ǘ�

};
struct RenderData {
	CUdeviceptr d_vertex = 0;
	CUdeviceptr d_normal = 0;
	CUdeviceptr d_texcoord = 0;
	CUdeviceptr d_textures = 0;
	CUdeviceptr d_instanceID = 0;
	CUdeviceptr d_insatnceData = 0;
	CUdeviceptr d_light_primID = 0;
	CUdeviceptr d_light_nee_weight = 0;
	CUdeviceptr d_light_color = 0;
	CUdeviceptr d_light_colorIndex = 0;

	RenderData() {}
	~RenderData() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertex)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_normal)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_texcoord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_textures)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instanceID)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_insatnceData)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_light_primID)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_light_nee_weight)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_light_color)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_light_colorIndex)));
	}
};


struct AccelationStructureData {
	std::vector<OptixTraversableHandle> gas_handle_array;
	std::vector<CUdeviceptr> d_gas_output_buffer_array;

	std::vector<OptixInstance> instance_array;
	OptixTraversableHandle ias_handle;

	CUdeviceptr d_ias_output_buffer;
	CUdeviceptr d_instance;
	OptixBuildInput instance_build = {};
	OptixAccelBuildOptions ias_accel_options = {};
	OptixAccelBufferSizes ias_buffer_sizes;
	CUdeviceptr d_temp_buffer_ias;

	std::vector<InsatanceData> instance_data;
	std::vector<unsigned int> face_instanceID;

	CUdeviceptr d_instance_data = 0;
	CUdeviceptr d_face_instanceID = 0;

	~AccelationStructureData() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ias_output_buffer)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instance)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_ias)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_instance_data)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_face_instanceID)));
		for (int i = 0; i < d_gas_output_buffer_array.size(); i++) {
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer_array[i])));
		}
	}

	void insatnceTransformInit() {
		const size_t instance_data_size = sizeof(InsatanceData) * instance_data.size();
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instance_data), instance_data_size));
		CUDA_CHECK(
			cudaMemcpy(
				reinterpret_cast<void*>(d_instance_data),
				instance_data.data(),
				instance_data_size,
				cudaMemcpyHostToDevice
			)
		);

		const size_t face_instanceID_size = sizeof(unsigned int) * face_instanceID.size();
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_instanceID), face_instanceID_size));
		CUDA_CHECK(
			cudaMemcpy(
				reinterpret_cast<void*>(d_face_instanceID),
				face_instanceID.data(),
				face_instanceID_size,
				cudaMemcpyHostToDevice
			)
		);
	}
	void instanceTransformUpdate() {
		const size_t instance_data_size = sizeof(InsatanceData) * instance_data.size();
		CUDA_CHECK(
			cudaMemcpy(
				reinterpret_cast<void*>(d_instance_data),
				instance_data.data(),
				instance_data_size,
				cudaMemcpyHostToDevice
			)
		);
	}
};

struct SceneData {
	std::vector<float3> vertices;
	std::vector<float3> normal;
	std::vector<float2> uv;
	std::vector<unsigned int> index;

	std::vector<Material> material;
	std::vector<unsigned int> material_index;

	std::vector<unsigned int> light_faceID;
	std::vector<float3> light_color;
	std::vector<unsigned int> light_colorIndex;
	std::vector<float> light_weight;
	std::vector<float> light_nee_weight;

	float3 directional_light_direction = normalize(make_float3(1, 1, 0));
	float3 directional_light_color = make_float3(1);
	float directional_light_weight;
	int direcitonal_light_animation = -1;

	std::vector<std::shared_ptr<Texture>> textures;
	std::vector<int> texture_index;

	std::shared_ptr<HDRTexture> ibl_texture = nullptr;
	float3 backGround;

	CameraStatus camera;

	std::vector<Animation> animation;
	std::vector<GASData> gas_data;

	void lightWeightUpData() {
		if (light_weight.size() == 0)return;
		float weight_sum = 0;
		for (int i = 0; i < light_weight.size(); i++) {
			weight_sum += light_weight[i];
		}

		light_nee_weight.resize(light_weight.size());
		light_nee_weight[0] = 0;
		for (int i = 1; i < light_weight.size(); i++) {
			light_nee_weight[i] = light_weight[i - 1] / weight_sum + light_nee_weight[i - 1];
		}
		/*
		Log::DebugLog(light_faceID);
		Log::DebugLog(light_weight);
		Log::DebugLog(light_nee_weight);
		Log::DebugLog(light_nee_weight.size());
		*/
	}
};

struct BufferObject {
	float4* buffer;
	unsigned int width;
	unsigned int height;
	CUdeviceptr d_gpu_buffer = 0;

	BufferObject(unsigned int in_width, unsigned int in_height) {
		width = in_width;
		height = in_height;

		buffer = new float4[in_width * in_height];

		const size_t buffer_size = sizeof(float4) * in_width * in_height;
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_gpu_buffer),
			buffer_size
		));
	}

	~BufferObject() {
		delete[] buffer;
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gpu_buffer)));
	}

	void cpyGPUBufferToHost() {
		const size_t buffer_size = sizeof(float4) * width * height;
		CUDA_CHECK(cudaMemcpy(
			buffer,
			reinterpret_cast<void*>(d_gpu_buffer),
			buffer_size,
			cudaMemcpyDeviceToHost
		));
	}
};

class Renderer {
private:
	OptixDeviceContext context = nullptr;

	//OptixTraversableHandle gas_handle;
	//CUdeviceptr d_gas_output_buffer;
	AccelationStructureData ac_data = {};

	OptixModule module = nullptr;
	OptixPipelineCompileOptions pipeline_compile_options = {};

	OptixProgramGroup raygen_prog_group = nullptr;
	OptixProgramGroup miss_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_occulusion = nullptr;

	OptixPipeline pipeline = nullptr;

	OptixShaderBindingTable sbt = {};

	std::vector<cudaArray_t> textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;

	cudaArray_t ibl_texture_array;
	cudaTextureObject_t ibl_texture_object;
	bool have_ibl = false;

	SceneData sceneData;
	RenderData renderData;

	unsigned int width, height;

	char log[2048]; // For error reporting from OptiX creation functions

	void contextInit() {
		// Initialize CUDA
		CUDA_CHECK(cudaFree(0));

		// Initialize the OptiX API, loading all API entry points
		OPTIX_CHECK(optixInit());

		// Specify context options
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;

		// Associate a CUDA context (and therefore a specific GPU) with this
		// device context
		CUcontext cuCtx = 0;  // zero means take the current context
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
	}
	void accelInit() {
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		for (int i = 0; i < sceneData.gas_data.size(); i++) {
			auto& gas_data = sceneData.gas_data[i];
			Log::DebugLog(gas_data.vert_offset);
			Log::DebugLog(gas_data.poly_n);
			Log::DebugLog(gas_data.animation_index);

			const size_t vertices_size = sizeof(float3) * gas_data.poly_n * 3;
			CUdeviceptr d_vertices = 0;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_vertices),
				sceneData.vertices.data() + gas_data.vert_offset,
				vertices_size,
				cudaMemcpyHostToDevice
			));

			const size_t mat_size = sizeof(uint32_t) * gas_data.poly_n;
			CUdeviceptr d_material = 0;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_material), mat_size));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_material),
				sceneData.material_index.data() + gas_data.vert_offset / 3,
				mat_size,
				cudaMemcpyHostToDevice
			));

			//the number of flags is equal to the number of Material
			std::vector<uint32_t> triangle_input_flags;
			triangle_input_flags.resize(sceneData.material.size());
			for (int i = 0; i < triangle_input_flags.size(); i++) {
				triangle_input_flags[i] = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
			}

			OptixBuildInput triangle_input = {};
			triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
			triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangle_input.triangleArray.numVertices = static_cast<uint32_t>(gas_data.poly_n * 3);
			triangle_input.triangleArray.vertexBuffers = &d_vertices;
			triangle_input.triangleArray.flags = triangle_input_flags.data();
			triangle_input.triangleArray.numSbtRecords = sceneData.material.size();
			triangle_input.triangleArray.sbtIndexOffsetBuffer = d_material;
			triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
			triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

			OptixAccelBufferSizes gas_buffer_sizes;
			OPTIX_CHECK(optixAccelComputeMemoryUsage(
				context,
				&accel_options,
				&triangle_input,
				1, // Number of build inputs
				&gas_buffer_sizes
			));

			CUdeviceptr d_temp_buffer_gas;
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_temp_buffer_gas),
				gas_buffer_sizes.tempSizeInBytes
			));

			CUdeviceptr d_gas_output_buffer;
			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&d_gas_output_buffer),
				gas_buffer_sizes.outputSizeInBytes
			));

			OptixTraversableHandle gas_handle;
			OPTIX_CHECK(optixAccelBuild(
				context,
				0,                  // CUDA stream
				&accel_options,
				&triangle_input,
				1,                  // num build inputs
				d_temp_buffer_gas,
				gas_buffer_sizes.tempSizeInBytes,
				d_gas_output_buffer,
				gas_buffer_sizes.outputSizeInBytes,
				&gas_handle,
				nullptr,            // emitted property list
				0                   // num emitted properties
			));

			ac_data.d_gas_output_buffer_array.push_back(d_gas_output_buffer);
			ac_data.gas_handle_array.push_back(gas_handle);
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_material)));

		}

		unsigned int sum_face = 0;
		//IAS construct
		for (int i = 0; i < ac_data.gas_handle_array.size(); i++) {
			OptixInstance instance = {};
			float transform[12] = { 1,0,0,0,0,1,0,0,0,0,1,0 };
			memcpy(instance.transform, transform, sizeof(float) * 12);
			instance.instanceId = i;
			instance.visibilityMask = 255;
			instance.sbtOffset = 0;
			instance.flags = OPTIX_INSTANCE_FLAG_NONE;
			instance.traversableHandle = ac_data.gas_handle_array[i];

			ac_data.instance_array.push_back(instance);
			InsatanceData instance_data;
			instance_data.faceIDoffset = sum_face;
			instance_data.instanceID = instance.instanceId;
			memcpy(instance_data.transform, transform, sizeof(float) * 12);

			Log::DebugLog(instance_data.faceIDoffset);
			Log::DebugLog(instance_data.instanceID);

			ac_data.instance_data.push_back(instance_data);
			for (int j = 0; j < sceneData.gas_data[i].poly_n; j++) {
				ac_data.face_instanceID.push_back(instance.instanceId);
			}

			sum_face += sceneData.gas_data[i].poly_n;
		}

		Log::DebugLog("insatnce ID equel sum face", sceneData.vertices.size() / 3 == ac_data.face_instanceID.size());
		Log::DebugLog("insatnce Data check", ac_data.instance_data.size() == ac_data.instance_array.size());
		ac_data.insatnceTransformInit();

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ac_data.d_instance), sizeof(OptixInstance) * ac_data.instance_array.size()));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ac_data.d_instance), ac_data.instance_array.data(), sizeof(OptixInstance) * ac_data.instance_array.size(), cudaMemcpyHostToDevice));

		ac_data.instance_build.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		ac_data.instance_build.instanceArray.instances = ac_data.d_instance;
		ac_data.instance_build.instanceArray.numInstances = static_cast<uint32_t>(ac_data.instance_array.size());

		ac_data.ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		ac_data.ias_accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			context,
			&ac_data.ias_accel_options,
			&ac_data.instance_build,
			1, // Number of build inputs
			&ac_data.ias_buffer_sizes
		));

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&ac_data.d_temp_buffer_ias),
			ac_data.ias_buffer_sizes.tempSizeInBytes
		));

		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&ac_data.d_ias_output_buffer),
			ac_data.ias_buffer_sizes.outputSizeInBytes
		));

		OPTIX_CHECK(optixAccelBuild(
			context,
			0,                  // CUDA stream
			&ac_data.ias_accel_options,
			&ac_data.instance_build,
			1,                  // num build inputs
			ac_data.d_temp_buffer_ias,
			ac_data.ias_buffer_sizes.tempSizeInBytes,
			ac_data.d_ias_output_buffer,
			ac_data.ias_buffer_sizes.outputSizeInBytes,
			&ac_data.ias_handle,
			nullptr,            // emitted property list
			0                   // num emitted properties
		));

		const size_t vertices_size = sizeof(float3) * sceneData.vertices.size();
		CUdeviceptr d_vertices = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_vertices),
			sceneData.vertices.data(),
			vertices_size,
			cudaMemcpyHostToDevice
		));

		const size_t normals_size = sizeof(float3) * sceneData.normal.size();
		CUdeviceptr d_normals = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_normals), normals_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_normals),
			sceneData.normal.data(),
			normals_size,
			cudaMemcpyHostToDevice
		));

		const size_t texcoords_size = sizeof(float2) * sceneData.uv.size();
		CUdeviceptr d_texcoords = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_texcoords), texcoords_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_texcoords),
			sceneData.uv.data(),
			texcoords_size,
			cudaMemcpyHostToDevice
		));

		const size_t light_faceID_size = sizeof(unsigned int) * sceneData.light_faceID.size();
		CUdeviceptr d_light_faceID = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_light_faceID), light_faceID_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_light_faceID),
			sceneData.light_faceID.data(),
			light_faceID_size,
			cudaMemcpyHostToDevice
		));

		const size_t light_nee_weight_size = sizeof(float) * sceneData.light_nee_weight.size();
		CUdeviceptr d_light_nee_weight = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_light_nee_weight), light_nee_weight_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_light_nee_weight),
			sceneData.light_nee_weight.data(),
			light_nee_weight_size,
			cudaMemcpyHostToDevice
		));

		const size_t light_color_size = sizeof(float3) * sceneData.light_color.size();
		CUdeviceptr d_light_color = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_light_color), light_color_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_light_color),
			sceneData.light_color.data(),
			light_color_size,
			cudaMemcpyHostToDevice
		));

		const size_t light_colorIndex_size = sizeof(unsigned int) * sceneData.light_colorIndex.size();
		CUdeviceptr d_light_colorIndex = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_light_colorIndex), light_colorIndex_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_light_colorIndex),
			sceneData.light_colorIndex.data(),
			light_colorIndex_size,
			cudaMemcpyHostToDevice
		));

		renderData.d_vertex = d_vertices;
		renderData.d_normal = d_normals;
		renderData.d_texcoord = d_texcoords;
		renderData.d_light_primID = d_light_faceID;
		renderData.d_light_nee_weight = d_light_nee_weight;
		renderData.d_light_color = d_light_color;
		renderData.d_light_colorIndex = d_light_colorIndex;
	}

	void IASUpdate(float time) {
		Log::DebugLog("IAS Updating");

		for (int i = 0; i < ac_data.instance_array.size(); i++) {
			Affine4x4 tf = sceneData.animation[sceneData.gas_data[i].animation_index].getAnimationAffine(time);
			float transform[12] = { tf[0],tf[1],tf[2],tf[3],tf[4],tf[5],tf[6],tf[7],tf[8],tf[9],tf[10],tf[11] };
			memcpy(ac_data.instance_array[i].transform, transform, sizeof(float) * 12);
			memcpy(ac_data.instance_data[i].transform, transform, sizeof(float) * 12);
		}

		ac_data.instanceTransformUpdate();

		ac_data.ias_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		ac_data.ias_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(ac_data.d_instance), ac_data.instance_array.data(), sizeof(OptixInstance) * ac_data.instance_array.size(), cudaMemcpyHostToDevice));
		OPTIX_CHECK(optixAccelBuild(
			context,
			0,                  // CUDA stream
			&ac_data.ias_accel_options,
			&ac_data.instance_build,
			1,                  // num build inputs
			ac_data.d_temp_buffer_ias,
			ac_data.ias_buffer_sizes.tempSizeInBytes,
			ac_data.d_ias_output_buffer,
			ac_data.ias_buffer_sizes.outputSizeInBytes,
			&ac_data.ias_handle,
			nullptr,            // emitted property list
			0                   // num emitted properties
		));

		//Light Weight Update
		for (int i = 0; i < sceneData.light_faceID.size(); i++) {
			unsigned int primitive_id = sceneData.light_faceID[i];
			unsigned int instance_id = ac_data.face_instanceID[primitive_id];
			Affine4x4 affine = sceneData.animation[sceneData.gas_data[instance_id].animation_index].getAnimationAffine(time);

			float3 v1 = affineConvertPoint(affine, sceneData.vertices[primitive_id * 3]);
			float3 v2 = affineConvertPoint(affine, sceneData.vertices[primitive_id * 3 + 1]);
			float3 v3 = affineConvertPoint(affine, sceneData.vertices[primitive_id * 3 + 2]);
			float Area = length(cross(v2 - v1, v3 - v1)) * 0.5f;

			float3 light_color = sceneData.light_color[sceneData.light_colorIndex[i]];
			float radiance = 0.2126 * light_color.x + 0.7152 * light_color.y + 0.0722 * light_color.z;
			sceneData.light_weight[i] = Area * radiance;
		}

		sceneData.lightWeightUpData();

		const size_t light_nee_weight_size = sizeof(float) * sceneData.light_nee_weight.size();

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(renderData.d_light_nee_weight),
			sceneData.light_nee_weight.data(),
			light_nee_weight_size,
			cudaMemcpyHostToDevice
		));
		Log::DebugLog("IAS Updating Finished");
	}

	void moduleInit() {
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

		pipeline_compile_options.usesMotionBlur = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.numPayloadValues = 3;
		pipeline_compile_options.numAttributeValues = 3;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
		pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		size_t      inputSize = 0;
		const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "myPathTracer.cu", inputSize);
		size_t sizeof_log = sizeof(log);

		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			context,
			&module_compile_options,
			&pipeline_compile_options,
			input,
			inputSize,
			log,
			&sizeof_log,
			&module
		));
	}

	void programInit() {
		OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

		OptixProgramGroupDesc raygen_prog_group_desc = {}; //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&raygen_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&raygen_prog_group
		));

		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&miss_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&miss_prog_group
		));

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		hitgroup_prog_group_desc.hitgroup.moduleAH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ch";

		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group
		));
		memset(&hitgroup_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occulusion";
		hitgroup_prog_group_desc.hitgroup.moduleAH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__occulusion";

		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_occulusion
		));
	}
	void pipelineInit() {
		const uint32_t    max_trace_depth = 1;
		OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = max_trace_depth;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixPipelineCreate(
			context,
			&pipeline_compile_options,
			&pipeline_link_options,
			program_groups,
			sizeof(program_groups) / sizeof(program_groups[0]),
			log,
			&sizeof_log,
			&pipeline
		));

		OptixStackSizes stack_sizes = {};
		for (auto& prog_group : program_groups)
		{
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			1  // maxTraversableDepth
		));
	}

	void stbInit() {

		CUdeviceptr  raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
		RayGenSbtRecord rg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(raygen_record),
			&rg_sbt,
			raygen_record_size,
			cudaMemcpyHostToDevice
		));

		CUdeviceptr miss_record;
		size_t      miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
		MissSbtRecord ms_sbt;
		ms_sbt.data.bg_color = sceneData.backGround;
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(miss_record),
			&ms_sbt,
			miss_record_size,
			cudaMemcpyHostToDevice
		));

		unsigned int MAT_COUNT = sceneData.material.size();
		CUdeviceptr d_hitgroup_records;
		const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_hitgroup_records),
			hitgroup_record_size * MAT_COUNT * RAY_TYPE
		));

		std::vector<HitGroupSbtRecord> hitgroup_record;
		hitgroup_record.resize(MAT_COUNT * RAY_TYPE);
		for (int i = 0; i < sceneData.material.size(); ++i) {
			{
				const int sbt_idx = i * RAY_TYPE;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hitgroup_record[sbt_idx]));

				Log::DebugLog(sceneData.material[i].material_name + " creating...");

				//Diffuse	
				hitgroup_record[sbt_idx].data.diffuse = sceneData.material[i].base_color;
				hitgroup_record[sbt_idx].data.diffuse_texID = sceneData.material[i].base_color_tex;

				//Specular
				hitgroup_record[sbt_idx].data.specular = sceneData.material[i].specular;
				hitgroup_record[sbt_idx].data.specular_texID = sceneData.material[i].specular_tex;

				//Roughness
				hitgroup_record[sbt_idx].data.roughness = sceneData.material[i].roughness;
				hitgroup_record[sbt_idx].data.roughness_texID = sceneData.material[i].roughness_tex;

				//Metallic
				hitgroup_record[sbt_idx].data.metallic = sceneData.material[i].metallic;
				hitgroup_record[sbt_idx].data.metallic_texID = sceneData.material[i].metallic_tex;

				//Sheen
				hitgroup_record[sbt_idx].data.sheen = sceneData.material[i].sheen;
				hitgroup_record[sbt_idx].data.sheen_texID = sceneData.material[i].sheen_tex;

				//Subsurface
				hitgroup_record[sbt_idx].data.subsurface = sceneData.material[i].subsurface;
				hitgroup_record[sbt_idx].data.subsurface_texID = sceneData.material[i].subsurface_tex;

				// Clearcoat
				hitgroup_record[sbt_idx].data.clearcoat = sceneData.material[i].clearcoat;
				hitgroup_record[sbt_idx].data.clearcoat_texID = sceneData.material[i].clearcoat_tex;

				//IOR
				hitgroup_record[sbt_idx].data.ior = sceneData.material[i].ior;

				//Transmission
				hitgroup_record[sbt_idx].data.transmission = sceneData.material[i].transmission;

				//NormalMap
				hitgroup_record[sbt_idx].data.normalmap_texID = sceneData.material[i].normal_tex;

				//BumpMap
				hitgroup_record[sbt_idx].data.bumpmap_texID = sceneData.material[i].bump_tex;

				//Emmision
				hitgroup_record[sbt_idx].data.emission_color = sceneData.material[i].emmision_color;
				hitgroup_record[sbt_idx].data.emission_texID = sceneData.material[i].emmision_color_tex;

				//Ideal Specular
				hitgroup_record[sbt_idx].data.is_idealSpecular = sceneData.material[i].ideal_specular;

				//Materila ID
				hitgroup_record[sbt_idx].data.MaterialID = i;
			}

			{
				const int sbt_idx = i * RAY_TYPE + 1;
				memset(&hitgroup_record[sbt_idx], 0, hitgroup_record_size);
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_occulusion, &hitgroup_record[sbt_idx]));
				//Diffuse	
				hitgroup_record[sbt_idx].data.diffuse = sceneData.material[i].base_color;
				hitgroup_record[sbt_idx].data.diffuse_texID = sceneData.material[i].base_color_tex;

				//Specular
				hitgroup_record[sbt_idx].data.specular = sceneData.material[i].specular;
				hitgroup_record[sbt_idx].data.specular_texID = sceneData.material[i].specular_tex;

				//Roughness
				hitgroup_record[sbt_idx].data.roughness = sceneData.material[i].roughness;
				hitgroup_record[sbt_idx].data.roughness_texID = sceneData.material[i].roughness_tex;

				//Metallic
				hitgroup_record[sbt_idx].data.metallic = sceneData.material[i].metallic;
				hitgroup_record[sbt_idx].data.metallic_texID = sceneData.material[i].metallic_tex;

				//Sheen
				hitgroup_record[sbt_idx].data.sheen = sceneData.material[i].sheen;
				hitgroup_record[sbt_idx].data.sheen_texID = sceneData.material[i].sheen_tex;

				//IOR
				hitgroup_record[sbt_idx].data.ior = sceneData.material[i].ior;

				//NormalMap
				hitgroup_record[sbt_idx].data.normalmap_texID = sceneData.material[i].normal_tex;

				//BumpMap
				hitgroup_record[sbt_idx].data.bumpmap_texID = sceneData.material[i].bump_tex;

				//Emmision
				hitgroup_record[sbt_idx].data.emission_color = sceneData.material[i].emmision_color;
				hitgroup_record[sbt_idx].data.emission_texID = sceneData.material[i].emmision_color_tex;

				//Ideal Specular
				hitgroup_record[sbt_idx].data.is_idealSpecular = sceneData.material[i].ideal_specular;

				//Materila ID
				hitgroup_record[sbt_idx].data.MaterialID = i;
			}
		}

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_hitgroup_records),
			hitgroup_record.data(),
			hitgroup_record_size * MAT_COUNT * RAY_TYPE,
			cudaMemcpyHostToDevice
		));

		sbt.raygenRecord = raygen_record;
		sbt.missRecordBase = miss_record;
		sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = d_hitgroup_records;
		sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
		sbt.hitgroupRecordCount = RAY_TYPE * MAT_COUNT;
	}

	void textureBind() {
		int numTextures = (int)sceneData.textures.size();

		textureArrays.resize(numTextures);
		textureObjects.resize(numTextures);

		for (int textureID = 0; textureID < numTextures; textureID++) {
			auto texture = sceneData.textures[textureID];
			Log::DebugLog("Texture ID ", textureID);
			Log::DebugLog("Texture ", texture->tex_name);
			Log::DebugLog("Texture Type", texture->tex_Type);
			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = texture->width;
			int32_t height = texture->height;
			int32_t numComponents = 4;
			int32_t pitch = width * numComponents * sizeof(uint8_t);
			channel_desc = cudaCreateChannelDesc<uchar4>();

			cudaArray_t& pixelArray = textureArrays[textureID];
			CUDA_CHECK(cudaMallocArray(&pixelArray,
				&channel_desc,
				width, height));

			CUDA_CHECK(cudaMemcpy2DToArray(pixelArray,
				/* offset */0, 0,
				texture->pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = pixelArray;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeNormalizedFloat;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
			tex_desc.sRGB = 1; //png Convert sRGB

			if (texture->tex_Type == "Normalmap") {
				tex_desc.sRGB = 0;
			}

			// Create texture object
			cudaTextureObject_t cuda_tex = 0;
			CUDA_CHECK(cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
			textureObjects[textureID] = cuda_tex;

		}

		const size_t texture_object_size = sizeof(cudaTextureObject_t) * textureObjects.size();
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&renderData.d_textures), texture_object_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(renderData.d_textures),
			textureObjects.data(),
			texture_object_size,
			cudaMemcpyHostToDevice
		));
		Log::DebugLog("Textures Loaded");

		Log::DebugLog("IBL texture Load");
		{
			auto texture = sceneData.ibl_texture;
			if (texture == nullptr) {
				texture = std::make_shared<HDRTexture>(sceneData.backGround);
			}
			cudaResourceDesc res_desc = {};

			cudaChannelFormatDesc channel_desc;
			int32_t width = texture->width;
			int32_t height = texture->height;
			int32_t pitch = width * sizeof(float4);
			channel_desc = cudaCreateChannelDesc<float4>();

			CUDA_CHECK(cudaMallocArray(&ibl_texture_array,
				&channel_desc,
				width, height));

			CUDA_CHECK(cudaMemcpy2DToArray(ibl_texture_array,
				0, 0,
				texture->pixel,
				pitch, pitch, height,
				cudaMemcpyHostToDevice));

			res_desc.resType = cudaResourceTypeArray;
			res_desc.res.array.array = ibl_texture_array;

			cudaTextureDesc tex_desc = {};
			tex_desc.addressMode[0] = cudaAddressModeWrap;
			tex_desc.addressMode[1] = cudaAddressModeWrap;
			tex_desc.filterMode = cudaFilterModeLinear;
			tex_desc.readMode = cudaReadModeElementType;
			tex_desc.normalizedCoords = 1;
			tex_desc.maxAnisotropy = 1;
			tex_desc.maxMipmapLevelClamp = 99;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.mipmapFilterMode = cudaFilterModePoint;
			tex_desc.borderColor[0] = 1.0f;
			tex_desc.sRGB = 0;

			CUDA_CHECK(cudaCreateTextureObject(&ibl_texture_object, &res_desc, &tex_desc, nullptr));

			have_ibl = true;
		}

	}
public:
	Renderer(unsigned int width, unsigned int height, const SceneData& sceneData) :
		width(width), height(height), sceneData(sceneData) {

	}


	~Renderer() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
		//CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
		for (int i = 0; i < textureArrays.size(); i++) {
			CUDA_CHECK(cudaFreeArray(textureArrays[i]));
		}
		for (int i = 0; i < textureObjects.size(); i++) {
			CUDA_CHECK(cudaDestroyTextureObject(textureObjects[i]));
		}
		CUDA_CHECK(cudaDestroyTextureObject(ibl_texture_object));
		CUDA_CHECK(cudaFreeArray(ibl_texture_array));

		OPTIX_CHECK(optixPipelineDestroy(pipeline));
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_occulusion));
		OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
		OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
		OPTIX_CHECK(optixModuleDestroy(module));

		OPTIX_CHECK(optixDeviceContextDestroy(context));
	}
	void build() {
		Log::StartLog("Context Initialize");
		contextInit();
		Log::EndLog("Context Initialize");

		Log::StartLog("Acceralation Structure Initialize");
		accelInit();
		Log::EndLog("Acceralation Structure Initialize");

		Log::StartLog("Module Initialize");
		moduleInit();
		Log::EndLog("Module Initialize");

		Log::StartLog("Program Initialize");
		programInit();
		Log::EndLog("Program Initialize");

		Log::StartLog("Pipeline Initialize");
		pipelineInit();
		Log::EndLog("Pipeline Initialize");

		Log::StartLog("Shading Binding Table Initialize");
		stbInit();
		textureBind();
		Log::EndLog("Shading Binding Table Initialize");
	}

	void render(unsigned int sampling, unsigned int RENDERMODE, const std::string& filename, CameraStatus& camera, float time) {
		/*
		float now_rendertime = time;

		CUstream stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		sutil::CUDAOutputBuffer<float4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
		sutil::CUDAOutputBuffer<float4> AOV_albedo(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
		sutil::CUDAOutputBuffer<float4> AOV_normal(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
		sutil::CUDAOutputBuffer<float4> denoiser_output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);

		sutil::ImageBuffer buffer;

		OptixDenoiserManager denoiser_manager(width,height,context,stream);

		long long animation_renderingTime = 0;

		for (int frame = 0; frame < 1; frame++) {
			auto start = std::chrono::system_clock::now();

			Log::StartLog("Rendering");
			Log::DebugLog("Sample", sampling);
			Log::DebugLog("Width", width);
			Log::DebugLog("Height", height);
			Log::DebugLog("Frame", frame);

			if (RENDERMODE == PATHTRACE) {
				Log::DebugLog("RenderMode", "PathTrace");
			}
			else if (RENDERMODE == NORMALCHECK) {
				Log::DebugLog("RenderMode", "NormalCheck");
			}
			else if (RENDERMODE == UVCHECK) {
				Log::DebugLog("RenderMode", "UVCheck");
			}
			else if (RENDERMODE == ALBEDOCHECK) {
				Log::DebugLog("RenderMode", "AlbedoCheck");
			}

			{

				IASUpdate(now_rendertime);

				sutil::Camera cam;
				configureCamera(cam, width, height);

				Params params;
				params.image = output_buffer.map();
				params.AOV_albedo = AOV_albedo.map();
				params.AOV_normal = AOV_normal.map();
				params.image_width = width;
				params.image_height = height;

				//params.handle = gas_handle;
				params.handle = ac_data.ias_handle;
				params.sampling = sampling;
				params.cam_eye = cam.eye();

				auto& cameraAnim = sceneData.animation[camera.cameraAnimationIndex];
				float4 camera_origin = cameraAnim.getTranslateAnimationAffine(now_rendertime) * make_float4(camera.origin, 1);
				float4 camera_direction = cameraAnim.getRotateAnimationAffine(now_rendertime) * make_float4(camera.direciton, 0);

				params.cam_ori = make_float3(camera_origin);
				params.cam_dir = normalize(make_float3(camera_direction)); //normalize(make_float3( - camera_origin));
				params.f = camera.f;

				params.instance_data = reinterpret_cast<InsatanceData*>(ac_data.d_instance_data);
				params.face_instanceID = reinterpret_cast<unsigned int*>(ac_data.d_face_instanceID);

				params.textures = reinterpret_cast<cudaTextureObject_t*>(renderData.d_textures);
				params.ibl = ibl_texture_object;
				params.has_ibl = have_ibl;

				params.normals = reinterpret_cast<float3*>(renderData.d_normal);
				params.texcoords = reinterpret_cast<float2*>(renderData.d_texcoord);
				params.vertices = reinterpret_cast<float3*> (renderData.d_vertex);

				params.light_nee_weight = reinterpret_cast<float*>(renderData.d_light_nee_weight);
				params.light_faceID = reinterpret_cast<unsigned int*>(renderData.d_light_primID);
				params.light_polyn = sceneData.light_faceID.size();
				params.light_color = reinterpret_cast<float3*>(renderData.d_light_color);
				params.light_colorIndex = reinterpret_cast<unsigned int*>(renderData.d_light_colorIndex);

				params.directional_light_direction = normalize(make_float3(0, 1, 0.2 * std::cos(3.14159256 * (now_rendertime / 10.0f))));
				params.directional_light_weight = sceneData.directional_light_weight;
				params.directional_light_color = sceneData.directional_light_color;

				params.frame = frame;

				params.RENDERE_MODE = RENDERMODE;
				cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);

				CUdeviceptr d_param;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(d_param),
					&params, sizeof(params),
					cudaMemcpyHostToDevice
				));


				OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, 1));
				CUDA_SYNC_CHECK();

				output_buffer.unmap();
			}

			//Denoiser
			{
				denoiser_manager.layerSet(
					AOV_albedo.map(),
					AOV_normal.map(),
					output_buffer.map(),
					denoiser_output_buffer.map()
				);

				denoiser_manager.denoise();
			}

			{
				buffer.data = denoiser_output_buffer.getHostPointer();
				if (RENDERMODE == NORMALCHECK) buffer.data = AOV_normal.getHostPointer();
				if (RENDERMODE == ALBEDOCHECK) buffer.data = AOV_albedo.getHostPointer();

				buffer.width = width;
				buffer.height = height;
				buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
				std::string imagename = filename + "_" + std::to_string(frame) + ".ppm";

				sutil::saveImage(imagename.c_str(), buffer, false);

				auto end = std::chrono::system_clock::now();
				auto Renderingtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				animation_renderingTime += Renderingtime;
				std::cout << std::endl << "Rendering finish" << std::endl;
				std::cout << "Rendering Time is " << Renderingtime << "ms" << std::endl;
				std::cout << std::endl << "----------------------" << std::endl;
				std::cout << "Rendering End" << std::endl;
				std::cout << "----------------------" << std::endl;
			}
		}
		std::cout << "Animation Rendering Time " << animation_renderingTime << "ms" << std::endl;
		*/
	}

	void animationRender(unsigned int sampling, const RenderType& render_type,
		const std::string& filename, CameraStatus& camera, FrameData flamedata, DenoiseType denoise_type) {

		float delta_rendertime = 1.0f / static_cast<float>(flamedata.fps);
		float now_rendertime = flamedata.minFrame * delta_rendertime;
		int renderIteration = flamedata.maxFrame - flamedata.minFrame;

		long long animation_renderingTime = 0;

		//CUDA stream
		CUstream stream;
		CUDA_CHECK(cudaStreamCreate(&stream));

		//Denoiser
		OptixDenoiserManager denoiser_manager(width, height, context, stream, denoise_type);

		//Output Image Buffer
		uchar4* output_data = new uchar4[width * height];
		sutil::ImageBuffer buffer;

		//Output Buffers
		BufferObject result_buffer(width, height);
		BufferObject albedo_buffer(width, height);
		BufferObject normal_buffer(width, height);
		BufferObject denoise_buffer(width, height);

		//Temporal Denoise
		BufferObject flow_buffer(width, height);
		BufferObject previous_buffer(width, height);

		//First Camera Frame
		float3 first_camera_origin = camera.origin;
		float3 first_camera_direction = camera.direciton;

		float3 pre_cam_origin = camera.origin;
		float3 pre_cam_dir = camera.direciton;

		if (camera.cameraAnimationIndex != -1) {
			//Camera Animation
			auto& firstcameraAnim = sceneData.animation[camera.cameraAnimationIndex];
			pre_cam_origin = make_float3(firstcameraAnim.getTranslateAnimationAffine(now_rendertime) * make_float4(camera.origin, 1));
			pre_cam_dir = normalize(make_float3(firstcameraAnim.getRotateAnimationAffine(now_rendertime) * make_float4(camera.direciton, 0)));
		}

		//Parametors
		Params params;
		CUdeviceptr d_param;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));

		for (int frame = flamedata.minFrame; frame < renderIteration + flamedata.minFrame; frame++) {
			auto start = std::chrono::system_clock::now();

			Log::StartLog("Rendering");
			Log::DebugLog("Sample", sampling);
			Log::DebugLog("Width", width);
			Log::DebugLog("Height", height);
			Log::DebugLog("Frame", frame);

			switch (render_type)
			{
			case PATHTRACE_NON_DENOISE:
				Log::DebugLog("PathTrace Non Denoise");
				break;
			case PATHTRACE_DENOISE:
				Log::DebugLog("PathTrace Denoise");
				break;
			case NORMAL:
				Log::DebugLog("NORMAL");
				break;
			case ALBEDO:
				Log::DebugLog("ALBEDO");
				break;
			default:
				Log::DebugLog("PathTrace Non Denoise");
				break;
			}


			//IAS and Light Weight Update
			{

				IASUpdate(now_rendertime);

			}

			//Parametor Update
			{


				//Output Buffers
				params.image = reinterpret_cast<float4*>(result_buffer.d_gpu_buffer);
				params.AOV_albedo = reinterpret_cast<float4*>(albedo_buffer.d_gpu_buffer);
				params.AOV_normal = reinterpret_cast<float4*>(normal_buffer.d_gpu_buffer);
				params.AOV_flow = reinterpret_cast<float4*>(flow_buffer.d_gpu_buffer);

				//Image status
				params.image_width = width;
				params.image_height = height;

				//Traversal Handle
				params.handle = ac_data.ias_handle;

				//Sampling count
				params.sampling = sampling;

				//Camera
				float3 camera_origin = camera.origin;
				float3 camera_direction = camera.direciton;

				if (camera.cameraAnimationIndex != -1) {
					//Camera Animation
					auto& cameraAnim = sceneData.animation[camera.cameraAnimationIndex];
					camera_origin = make_float3(cameraAnim.getTranslateAnimationAffine(now_rendertime) * make_float4(camera.origin, 1));
					camera_direction = normalize(make_float3(cameraAnim.getRotateAnimationAffine(now_rendertime) * make_float4(camera.direciton, 0)));
				}

				params.cam_ori = camera_origin;
				params.cam_dir = camera_direction;
				params.f = camera.f;

				//previous frame Camera status
				params.pre_cam_ori = pre_cam_origin;
				params.pre_cam_dir = pre_cam_dir;
				params.pre_f = camera.f;

				Log::DebugLog("cam_dir", params.cam_dir);
				Log::DebugLog("cam_ori", params.cam_ori);

				Log::DebugLog("pre_cam_dir", params.pre_cam_dir);
				Log::DebugLog("pre_cam_ori", params.pre_cam_ori);

				pre_cam_origin = params.cam_ori;
				pre_cam_dir = params.cam_dir;

				//Face and Instance , Texture Data
				params.instance_data = reinterpret_cast<InsatanceData*>(ac_data.d_instance_data);
				params.face_instanceID = reinterpret_cast<unsigned int*>(ac_data.d_face_instanceID);
				params.textures = reinterpret_cast<cudaTextureObject_t*>(renderData.d_textures);

				//IBL status
				params.ibl = ibl_texture_object;
				params.has_ibl = have_ibl;

				//Attribute Data
				params.normals = reinterpret_cast<float3*>(renderData.d_normal);
				params.texcoords = reinterpret_cast<float2*>(renderData.d_texcoord);
				params.vertices = reinterpret_cast<float3*> (renderData.d_vertex);

				//Light Data
				params.light_nee_weight = reinterpret_cast<float*>(renderData.d_light_nee_weight);
				params.light_faceID = reinterpret_cast<unsigned int*>(renderData.d_light_primID);
				params.light_polyn = sceneData.light_faceID.size();
				params.light_color = reinterpret_cast<float3*>(renderData.d_light_color);
				params.light_colorIndex = reinterpret_cast<unsigned int*>(renderData.d_light_colorIndex);

				//Directional Light Data
				float3 directional_light_direction = sceneData.directional_light_direction;
				if (sceneData.direcitonal_light_animation != -1) {
					auto& anim = sceneData.animation[sceneData.direcitonal_light_animation];
					directional_light_direction = normalize(make_float3(anim.getRotateAnimationAffine(now_rendertime) * make_float4(0,0,1,0)));
				}
				params.directional_light_direction = directional_light_direction;
				params.directional_light_weight = sceneData.directional_light_weight;
				params.directional_light_color = sceneData.directional_light_color;

				//Frame 
				params.frame = frame;


				//Parametor cpy to Device
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(d_param),
					&params, sizeof(params),
					cudaMemcpyHostToDevice
				));

				//Rendering Start
				OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
				CUDA_SYNC_CHECK();

			}

			//Denoise
			{
				if (render_type == RenderType::PATHTRACE_DENOISE)
				{
					//Layer Setting
					denoiser_manager.layerSet(
						reinterpret_cast<float4*>(albedo_buffer.d_gpu_buffer),
						reinterpret_cast<float4*>(normal_buffer.d_gpu_buffer),
						reinterpret_cast<float4*>(flow_buffer.d_gpu_buffer),
						reinterpret_cast<float4*>(result_buffer.d_gpu_buffer),
						reinterpret_cast<float4*>(denoise_buffer.d_gpu_buffer),
						reinterpret_cast<float4*>(previous_buffer.d_gpu_buffer)
					);

					//Denoise
					denoiser_manager.denoise();
				}

			}

			{
				//Output Buffer Image Pointer
				float4* data_pointer;

				switch (render_type)
				{
				case PATHTRACE_NON_DENOISE:
					result_buffer.cpyGPUBufferToHost();
					data_pointer = result_buffer.buffer;
					break;

				case PATHTRACE_DENOISE:
					denoise_buffer.cpyGPUBufferToHost();
					data_pointer = denoise_buffer.buffer;
					//flow_buffer.cpyGPUBufferToHost();
					//data_pointer = flow_buffer.buffer;
					break;

				case NORMAL:
					normal_buffer.cpyGPUBufferToHost();
					data_pointer = normal_buffer.buffer;
					break;

				case ALBEDO:
					albedo_buffer.cpyGPUBufferToHost();
					data_pointer = albedo_buffer.buffer;
					break;

				default:
					denoise_buffer.cpyGPUBufferToHost();
					data_pointer = denoise_buffer.buffer;
					break;
				}

				//Convert
				float4ConvertColor(data_pointer, output_data, width, height);

				dim3 grid(10, 10);
				dim3 block(4, 4, 1);

				
				//Buffer Setting
				buffer.data = output_data;
				buffer.width = width;
				buffer.height = height;
				buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
				std::string imagename = filename + "_" + std::to_string(frame) + ".png";

				//Save
				sutil::saveImage(imagename.c_str(), buffer, false);

				//Rendering Data
				auto end = std::chrono::system_clock::now();
				auto Renderingtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				animation_renderingTime += Renderingtime;
				std::cout << std::endl << "Rendering finish" << std::endl;
				std::cout << "Rendering Time is " << Renderingtime << "ms" << std::endl;
				std::cout << std::endl << "----------------------" << std::endl;
				std::cout << "Rendering End" << std::endl;
				std::cout << "----------------------" << std::endl;

				//Buffer Swap
				auto copy = previous_buffer.d_gpu_buffer;
				previous_buffer.d_gpu_buffer = denoise_buffer.d_gpu_buffer;
				denoise_buffer.d_gpu_buffer = copy;
			}
			now_rendertime += delta_rendertime;
		}

		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
		delete[] output_data;
		std::cout << "Animation Rendering Time " << animation_renderingTime << "ms" << std::endl;
	}
};

