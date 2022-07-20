#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include <fstream>
#include <iostream>
#include <string>
#include <myPathTracer/debugLog.h>
#include <external/stb/stb_image.h>

class Texture {
public:
	std::string tex_name;
	unsigned int width, height;
	uint32_t* pixel;

	Texture(const std::string& filename) {
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
	

