#pragma once
#include <optix.h>
#include <sutil/sutil.h>
#include <sutil/Exception.h>

static OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, float4* data)
{
	OptixImage2D oi;

	oi.width = width;
	oi.height = height;
	oi.rowStrideInBytes = width * sizeof(float4);
	oi.pixelStrideInBytes = sizeof(float4);
	oi.data = reinterpret_cast<CUdeviceptr>(data);
	oi.format = OPTIX_PIXEL_FORMAT_FLOAT4;

	return oi;
}

static OptixImage2D createOptixImage2D(unsigned int width, unsigned int height, uchar4* data)
{
	OptixImage2D oi;

	oi.width = width;
	oi.height = height;
	oi.rowStrideInBytes = width * sizeof(uchar4);
	oi.pixelStrideInBytes = sizeof(uchar4);
	oi.data = reinterpret_cast<CUdeviceptr>(data);
	oi.format = OPTIX_PIXEL_FORMAT_UCHAR4;

	return oi;
}

class OptixDenoiserManager {
private:
	OptixDeviceContext context = nullptr;
	OptixDenoiser denoiser = nullptr;
	CUstream cu_stream = nullptr;

	CUdeviceptr m_scratch = 0;
	uint32_t m_scratch_size = 0;
	CUdeviceptr m_state = 0;
	uint32_t m_state_size = 0;

	unsigned int width = 0;
	unsigned int height = 0;

	float4* albedo;
	float4* normal;
	float4* input;
	float4* output;
	float4* previous;
	float4* flow;

public:

	OptixDenoiserManager(const unsigned int& width, const unsigned int& height,
		OptixDeviceContext context, CUstream cu_stream) : width(width), height(height), context(context), cu_stream(cu_stream) {

		OptixDenoiserOptions options;
		options.guideAlbedo = 1;
		options.guideNormal = 1;

		OptixDenoiserModelKind model_kind;
		model_kind = OPTIX_DENOISER_MODEL_KIND_LDR;
		//model_kind = OPTIX_DENOISER_MODEL_KIND_TEMPORAL;


		OPTIX_CHECK(optixDenoiserCreate(
			context,
			model_kind,
			&options,
			&denoiser
		));

		//Setup
		{
			OptixDenoiserSizes denoiser_size;
			OPTIX_CHECK(optixDenoiserComputeMemoryResources(
				denoiser,
				width,
				height,
				&denoiser_size
			));

			m_scratch_size = static_cast<uint32_t>(denoiser_size.withOverlapScratchSizeInBytes);
			m_state_size = static_cast<uint32_t>(denoiser_size.stateSizeInBytes);

			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&m_scratch),
				m_scratch_size
			));

			CUDA_CHECK(cudaMalloc(
				reinterpret_cast<void**>(&m_state),
				m_state_size
			));

			OPTIX_CHECK(optixDenoiserSetup(
				denoiser,
				cu_stream,
				width,
				height,
				m_state,
				m_state_size,
				m_scratch,
				m_scratch_size
			));
		}
	}

	void layerSet(float4* in_albedo,float4* in_normal,float4* in_input,float4* in_output,float4* in_previous) {
		albedo = in_albedo;
		normal = in_normal;
		input = in_input;
		output = in_output;
		//previous = in_previous;
	}

	void denoise() {
		OptixDenoiserGuideLayer guidelayer;
		guidelayer.albedo = createOptixImage2D(width,height,albedo);
		guidelayer.normal = createOptixImage2D(width, height, normal);
		//guidelayer.flow = createOptixImage2D(width,height,flow);

		OptixDenoiserLayer layers;
		layers.input = createOptixImage2D(width, height, input);
		//layers.previousOutput = createOptixImage2D(width, height, input);
		layers.output = createOptixImage2D(width, height, output);

		OptixDenoiserParams param;
		param.denoiseAlpha = 0;
		param.blendFactor = 0;
		param.hdrAverageColor = 0;
		param.hdrIntensity = 0;
		
		OPTIX_CHECK(optixDenoiserInvoke(
			denoiser,
			cu_stream,
			&param,
			m_state,
			m_state_size,
			&guidelayer,
			&layers,
			1,
			0,
			0,
			m_scratch,
			m_scratch_size
		));
	}

	~OptixDenoiserManager() {
		OPTIX_CHECK(optixDenoiserDestroy(denoiser));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_scratch)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state)));
	}
};
