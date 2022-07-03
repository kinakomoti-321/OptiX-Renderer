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

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#include <myPathTracer/myPathTracer.h>
#include <myPathTracer/debugLog.h>
#include <myPathTracer/material.h>


template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;


void configureCamera(sutil::Camera& cam, const uint32_t width, const uint32_t height)
{
	cam.setEye({ 0.0f, 0.0f, 5.0f });
	cam.setLookat({ 0.0f, 0.0f, 0.0f });
	cam.setUp({ 0.0f, 1.0f, 3.0f });
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

static void addBox(std::vector<float3>& vert) {
	//Bottom
	vert.push_back({ 1.0f,-1.0f,1.0f });
	vert.push_back({ 1.0f,-1.0f,-1.0f });
	vert.push_back({ -1.0f,-1.0f,-1.0f });
	vert.push_back({ 1.0f,-1.0f,1.0f });
	vert.push_back({ -1.0f,-1.0f,-1.0f });
	vert.push_back({ -1.0f,-1.0f,1.0f });

	//Top
	vert.push_back({ 1.0f,1.0f,1.0f });
	vert.push_back({ 1.0f,1.0f,-1.0f });
	vert.push_back({ -1.0f,1.0f,-1.0f });
	vert.push_back({ 1.0f,1.0f,1.0f });
	vert.push_back({ -1.0f,1.0f,-1.0f });
	vert.push_back({ -1.0f,1.0f,1.0f });

	//Left
	vert.push_back({ -1.0f,1.0f,1.0f });
	vert.push_back({ -1.0f,-1.0f,1.0f });
	vert.push_back({ -1.0f,-1.0f,-1.0f });
	vert.push_back({ -1.0f,1.0f,1.0f });
	vert.push_back({ -1.0f,-1.0f,-1.0f });
	vert.push_back({ -1.0f,1.0f,-1.0f });

	//Right
	vert.push_back({ 1.0f,1.0f,1.0f });
	vert.push_back({ 1.0f,-1.0f,1.0f });
	vert.push_back({ 1.0f,-1.0f,-1.0f });
	vert.push_back({ 1.0f,1.0f,1.0f });
	vert.push_back({ 1.0f,-1.0f,-1.0f });
	vert.push_back({ 1.0f,1.0f,-1.0f });

	//Back
	vert.push_back({ -1.0f,1.0f,-1.0f });
	vert.push_back({ -1.0f,-1.0f,-1.0f });
	vert.push_back({ 1.0f,-1.0f,-1.0f });
	vert.push_back({ -1.0f,1.0f,-1.0f });
	vert.push_back({ 1.0f,-1.0f,-1.0f });
	vert.push_back({ 1.0f,1.0f,-1.0f });


	//Light
	vert.push_back({ 0.5f,0.99f,0.5f });
	vert.push_back({ 0.5f,.99f,-0.5f });
	vert.push_back({ -0.5f,.99f,-0.5f });
	vert.push_back({ 0.5f,.99f,0.5f });
	vert.push_back({ -0.5f,.99f,-0.5f });
	vert.push_back({ -0.5f,.99f,0.5f });
}


const unsigned int MAT_COUNT = 4;

struct RendererState {
	//Frame�Ȃǂ̊Ǘ�

};
struct RenderData {
	CUdeviceptr d_vertex = 0;
	CUdeviceptr d_normal = 0;
	CUdeviceptr d_uv = 0;

	RenderData() {}
	~RenderData() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertex)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_normal)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_uv)));
	}
};

struct SceneData {
	std::vector<float3> vertices;
	std::vector<float3> normal;
	std::vector<float2> uv;
	std::vector<unsigned int> material_index;
	std::vector<Material> material;
	

	float3 backGround;
};

class Renderer {
private:
	OptixDeviceContext context = nullptr;

	OptixTraversableHandle gas_handle;
	CUdeviceptr d_gas_output_buffer;

	OptixModule module = nullptr;
	OptixPipelineCompileOptions pipeline_compile_options = {};

	OptixProgramGroup raygen_prog_group = nullptr;
	OptixProgramGroup miss_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_occulusion = nullptr;

	OptixPipeline pipeline = nullptr;

	OptixShaderBindingTable sbt = {};

	SceneData sceneData;
	RenderData renderData;

	unsigned int width, height;
	std::string filename;

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
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// Triangle build input: simple list of three vertices

		const size_t vertices_size = sizeof(float3) * sceneData.vertices.size();
		CUdeviceptr d_vertices = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_vertices),
			sceneData.vertices.data(),
			vertices_size,
			cudaMemcpyHostToDevice
		));

		const size_t mat_size = sceneData.material_index.size() * sizeof(uint32_t);
		CUdeviceptr d_material = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_material), mat_size));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_material),
			sceneData.material_index.data(),
			mat_size,
			cudaMemcpyHostToDevice
		));


		// Our build input is a simple list of non-indexed triangle vertices
		const uint32_t triangle_input_flags[4] = {
			OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
			OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
			OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
			OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
		};
		OptixBuildInput triangle_input = {};
		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.numVertices = static_cast<uint32_t>(sceneData.vertices.size());
		triangle_input.triangleArray.vertexBuffers = &d_vertices;
		triangle_input.triangleArray.flags = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = MAT_COUNT;
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
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_gas_output_buffer),
			gas_buffer_sizes.outputSizeInBytes
		));

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

		// We can now free the scratch space buffer used during build and the vertex
		// inputs, since they are not needed by our trivial shading method
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
		//CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_material)));

		renderData.d_vertex = d_vertices;
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

		/*
		CUdeviceptr hitgroup_record;
		size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
		HitGroupSbtRecord hg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(hitgroup_record),
			&hg_sbt,
			hitgroup_record_size,
			cudaMemcpyHostToDevice
		));
		*/

		CUdeviceptr d_hitgroup_records;
		const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_hitgroup_records),
			hitgroup_record_size * MAT_COUNT * RAY_TYPE
		));

		HitGroupSbtRecord hitgroup_record[RAY_TYPE * MAT_COUNT];
		for (int i = 0; i < sceneData.material.size(); ++i) {
			{

				const int sbt_idx = i * RAY_TYPE;
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hitgroup_record[sbt_idx]));
				hitgroup_record[sbt_idx].data.emission_color = sceneData.material[i].emmision_color;
				hitgroup_record[sbt_idx].data.base_color = sceneData.material[i].base_color;
				hitgroup_record[sbt_idx].data.vertices = reinterpret_cast<float3*>(renderData.d_vertex);
			}

			{

				const int sbt_idx = i * RAY_TYPE + 1;
				memset(&hitgroup_record[sbt_idx], 0, hitgroup_record_size);
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_occulusion, &hitgroup_record[sbt_idx]));
			}
		}

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_hitgroup_records),
			hitgroup_record,
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
public:
	Renderer(unsigned int width,unsigned int height,const SceneData& sceneData,const std::string& filename):
		width(width),height(height),sceneData(sceneData), filename(filename) {

	}


	~Renderer() {
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));

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
		Log::EndLog("Shading Binding Table Initialize");
	}

	void render(unsigned int sampling) {
		sutil::CUDAOutputBuffer<uchar4> output_buffer(sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
		//
		// launch
		//
		{
			CUstream stream;
			CUDA_CHECK(cudaStreamCreate(&stream));

			sutil::Camera cam;
			configureCamera(cam, width, height);

			Params params;
			params.image = output_buffer.map();
			params.image_width = width;
			params.image_height = height;
			params.handle = gas_handle;
			params.sampling = sampling;
			params.cam_eye = cam.eye();
			cam.UVWFrame(params.cam_u, params.cam_v, params.cam_w);
			std::cout << width << " : " << height << std::endl;

			CUdeviceptr d_param;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_param),
				&params, sizeof(params),
				cudaMemcpyHostToDevice
			));
			OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
			CUDA_SYNC_CHECK();

			output_buffer.unmap();
		}

		//
		// Display results
		//
		{
			sutil::ImageBuffer buffer;
			buffer.data = output_buffer.getHostPointer();
			buffer.width = width;
			buffer.height = height;
			buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
			std::string imagename = filename + ".png";
			sutil::displayBufferWindow("RayTracing", buffer);
			sutil::saveImage(imagename.c_str(), buffer, false);
		}
	}
};

