#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <myPathTracer/debugLog.h>
#include <algorithm>
#include <external/tinygltf/stb_image.h>

class Texture {
public:
	std::string tex_name;
	std::string tex_Type;
	unsigned int width, height;
	uint32_t* pixel;

	Texture(const std::string& filename,const std::string& tex_Type):tex_Type(tex_Type) {
		tex_name = filename;
		int resx, resy;
		int comp;
		unsigned char* image = stbi_load(filename.c_str(), &resx, &resy, &comp, STBI_rgb_alpha);

		if (image) {
			width = resx;
			height = resy;
			pixel = (uint32_t*)image;
			Log::DebugLog(filename + " LOADED");
		}
		else {
			Log::DebugLog(filename + " NOT FOUND");
		}
	}
};
	

class HDRTexture {
public:
	std::string tex_name;
	std::string tex_Type;
	unsigned int width, height;
	float4* pixel;

	HDRTexture(const float3& background) {
		tex_name = "_background";
		tex_Type = "hdr";
		width = 1;
		height = 1;
		pixel = new float4[1];
		pixel[0] = make_float4(background, 0.0);
	}
	HDRTexture(const std::string& filename,const std::string& tex_Type):tex_Type(tex_Type) {
		tex_name = filename;
		int resx, resy;
		int comp;
		float* image = stbi_loadf(filename.c_str(), &resx, &resy, &comp, 0);

		Log::DebugLog("comp",comp);

		if (image) {
			width = resx;
			height = resy;

			pixel = new float4[height * width];

			for (int j = 0; j < height; j++) {
				for (int i = 0; i < width; i++) {
					int idx = i * 3 + j * 3 * width;
					pixel[i + width * j] = make_float4(image[idx], image[idx + 1], image[idx + 2],0.0);
				}
			}
			Log::DebugLog(filename + " LOADED");
		}
		else {
			Log::DebugLog(filename + " NOT FOUND");
		}
	}

	void writePNG(std::string filename)
	{
		std::ofstream file(filename + ".ppm");
		if (!file)
		{
			std::cerr << "failed to open " << filename << std::endl;
			return;
		}

		file << "P3" << std::endl;
		file << width << " " << height << std::endl;
		file << "255" << std::endl;
		for (int j = 0; j < height; ++j)
		{
			for (int i = 0; i < width; ++i)
			{
				const int idx = i + width * j;
				float R = pixel[idx].x;
				float G = pixel[idx].y;
				float B = pixel[idx].z;

				// Še¬•ª‚ð[0, 255]‚ÉŠÜ‚Ü‚ê‚é‚æ‚¤‚É•ÏŠ·‚µo—Í
				file << static_cast<unsigned int>(clamp(255.0f * R, 0.0f, 255.0f))
					<< " ";
				file << static_cast<unsigned int>(clamp(255.0f * G, 0.0f, 255.0f))
					<< " ";
				file << static_cast<unsigned int>(clamp(255.0f * B, 0.0f, 255.0f))
					<< std::endl;
			}
		}
		file.close();
		Log::DebugLog("Image_Writed");
	}

	~HDRTexture() {
		delete[] pixel;
	}
};
	

