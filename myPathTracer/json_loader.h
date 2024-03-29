#pragma once

#include <external/json/include/nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <myPathTracer/myPathTracer.h>
#include <myPathTracer/Denoiser.h>

struct SceneInformation {
	unsigned int width = 1;
	unsigned int height = 1;
	unsigned int sampling = 1;

	std::string gltf_filename;
	std::string gltf_filepath;

	bool use_hdr = false;
	float3 default_enviroment = { 0,0,0 };
	std::string hdrpath;

	float3 directional_light_color = { 0,0,0 };
	float directional_light_weight = 0.0;
	float3 directional_light_direction = { 0,1,0 };
	
	float3 camera_origin = { 0,0,0 };
	float3 camera_dir = { 1,0,0 };
	float camera_f = 1.0;

	bool use_time = true;
	std::string output_filename = "default";

	bool use_animation = true;
	unsigned int fps = 0;
	unsigned int minframe = 0;
	unsigned int maxframe = 1;

	RenderType render_type = RenderType::PATHTRACE_DENOISE;
	DenoiseType denoise_type = DenoiseType::NONE;
	
	float timelimit = 9.5;
};

bool loadSceneFile(std::string& scene_filename, SceneInformation& scene_info)
{
	try {
		scene_filename = "./scene_file.json";
		std::ifstream ifs(scene_filename);
		std::string jsonstr;

		if (ifs.fail()) {
			std::cout << "File " << scene_filename << " not found" << std::endl;
			return false;
		}
		std::string str;
		while (std::getline(ifs, str)) {
			jsonstr += str + "\n";
		}

		std::cout << jsonstr << std::endl;
		auto jobj = json::parse(jsonstr.c_str());

		scene_info.width = jobj["width"];
		scene_info.height = jobj["height"];
		scene_info.sampling = jobj["sampling"];
		scene_info.gltf_filename = jobj["gltf_filename"];
		scene_info.gltf_filepath = jobj["gltf_filepath"];

		scene_info.use_hdr = jobj["HDRI"]["use_hdr"];
		auto enviroment = jobj["HDRI"]["default_enviroment"];
		scene_info.default_enviroment =
			make_float3(enviroment[0], enviroment[1], enviroment[2]);
		scene_info.hdrpath = jobj["HDRI"]["HDRpath"];

		auto light_color = jobj["DirectionalLight"]["color"];
		scene_info.directional_light_color =
			make_float3(light_color[0], light_color[1], light_color[2]);
		scene_info.directional_light_weight = jobj["DirectionalLight"]["weight"];
		auto light_dir = jobj["DirectionalLight"]["direction"];
		scene_info.directional_light_direction =
			normalize(make_float3(light_dir[0], light_dir[1], light_dir[2]));

		scene_info.output_filename = jobj["Output"]["output_filename"];
		scene_info.use_time = jobj["Output"]["use_time"];

		scene_info.use_animation = jobj["FrameRate"]["use_animation"];
		scene_info.fps = jobj["FrameRate"]["fps"];
		scene_info.minframe = jobj["FrameRate"]["minframe"];
		scene_info.maxframe = jobj["FrameRate"]["maxframe"];

		std::string render_type = jobj["RenderType"];
		if (render_type == "PATHTRACE_DENOISE") {
			scene_info.render_type = RenderType::PATHTRACE_DENOISE;
		}
		else if (render_type == "PATHTRACE_NON_DENOISE") {
			scene_info.render_type = RenderType::PATHTRACE_NON_DENOISE;
		}
		else if (render_type == "ALBEDO") {
			scene_info.render_type = RenderType::ALBEDO;
		}
		else if (render_type == "NORMAL") {
			scene_info.render_type = RenderType::NORMAL;
		}
		else if (render_type == "FLOW") {
			scene_info.render_type = RenderType::FLOW;
		}
		else {
			scene_info.render_type = RenderType::PATHTRACE_DENOISE;
		}

		std::string denoise_type = jobj["DenoiseType"];
		if (render_type == "NONE") {
			scene_info.denoise_type = DenoiseType::NONE;
		}
		else if (render_type == "TEMPORAL") {
			scene_info.denoise_type = DenoiseType::TEMPORAL;
		}

		if (jobj["SaveSceneFile"]) {
			auto now = std::chrono::system_clock::now();
			std::time_t end_time = std::chrono::system_clock::to_time_t(now);
			std::string time = std::ctime(&end_time);

			auto start = std::chrono::system_clock::now();
			time.erase(std::remove(time.begin(), time.end(), ':'), time.end());
			time.erase(std::remove(time.begin(), time.end(), '\n'), time.end());

			std::ofstream file("scene_file" + time + ".json");
			file << jsonstr;
			file.close();

			std::cout << "Scene File Save " << "scene_file" + time + ".json" << std::endl;
		}

		auto camera_origin_array = jobj["Camera"]["Origin"];
		float3 camera_origin = make_float3(camera_origin_array[0],camera_origin_array[1],camera_origin_array[2]);

		auto camera_atlook_array = jobj["Camera"]["Atlook"];
		float3 camera_direction = normalize(make_float3(camera_atlook_array[0],camera_atlook_array[1],camera_atlook_array[2]) - camera_origin);

		scene_info.camera_origin = camera_origin;
		scene_info.camera_dir = camera_direction;
		scene_info.camera_f = jobj["Camera"]["F"];

		scene_info.timelimit = jobj["TimeLimit"];
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return false;
	}
	return true;
}

bool fpsLoader(unsigned int& fps) {
	std::string filename = "./fps.txt";
	try {
		std::ifstream ifs(filename);

		if (ifs.fail()) {
			std::cout << "File " << filename << " not found" << std::endl;
			return false;
		}
		std::string str;
		while (std::getline(ifs, str)) {
			fps = std::stoi(str);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "Caught exception: " << e.what() << "\n";
		return false;
	}
	return true;
}

std::ostream& operator<<(std::ostream& stream, const SceneInformation& info)
{
	stream << "width : " << info.width << std::endl;
	stream << "height : " << info.height << std::endl;
	stream << "sampling : " << info.sampling << std::endl;

	stream << "gltf_filename : " << info.gltf_filename << std::endl;
	stream << "gltf_filepath : " << info.gltf_filepath << std::endl;

	stream << "use_hdr : " << info.use_hdr << std::endl;
	stream << "default enviroment : " << info.default_enviroment << std::endl;
	stream << "HDRpath : " << info.hdrpath << std::endl;

	stream << "Light color : " << info.directional_light_color << std::endl;
	stream << "Light weight : " << info.directional_light_weight << std::endl;
	stream << "Light direction : " << info.directional_light_direction << std::endl;

	stream << "Output Filename : " << info.output_filename << std::endl;
	stream << "use Time : " << info.use_time << std::endl;

	stream << "fps : " << info.fps << std::endl;
	stream << "minFrame : " << info.minframe << std::endl;
	stream << "maxFrame : " << info.maxframe << std::endl;

	stream << "RenderType : " << info.render_type << std::endl;

	return stream;
}

/* Format Example
{
	"width": 1024,
	"height": 768,
	"sampling": 100,
	"gltf_filename": "glTF_test.gltf",
	"gltf_filepath": "./model/GLTF/San_Miguel/GLTF",
	"HDRI": {
		"use_hdr": true,
		"default_enviroment": [
			1,
			1,
			1
		],
		"HDRpath": "model/hdr/test.hdr"
	},
	"DirectionalLight": {
		"color": [
			1,
			1,
			1
		],
		"weight": 0.5,
		"direction": [
			0,
			1,
			0
		]
	},
	"Output": {
		"output_filename": "output_test",
		"use_time": false
	},
	"FrameRate": {
		"use_animation": true,
		"fps": 0,
		"minframe": 0,
		"maxframe": 1
	}
}
*/
