#pragma once

struct Material
{
	//Material Name
	std::string material_name;

	//Diffuse
	float3 base_color;
	int base_color_tex;

	//Specular
	float3 specular;
	int specular_tex;

	//Roughenss
	float roughness;
	int roughness_tex;

	//Metallic
	float metallic;
	int metallic_tex;

	//Sheen
	float sheen;
	int sheen_tex;

	//IOR
	float ior;

	//Normalmap
	int normal_tex;

	//BumpMap
	int bump_tex;

	//Emmision
	float3 emmision_color;
	int emmision_color_tex;	

	//Ideal Specular
	bool ideal_specular = false;
};


std::ostream& operator<<(std::ostream& stream, const Material& f)
{
	stream << "diffuse " << f.base_color << " texID : " << f.base_color_tex;
	stream << "specular " << f.specular << " texID : " << f.specular_tex;
	stream << "roughness " << f.roughness << " texID : " << f.roughness_tex;
	stream << "metallic " << f.metallic << " texID : " << f.metallic_tex;
	stream << "sheen " << f.sheen << " texID : " << f.sheen_tex;
	stream << "ior " << f.ior << " texID : " << f.sheen_tex;
	stream << "emission " << f.emmision_color << " texID : " << f.emmision_color_tex;
	
	return stream;
}
