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

	//Subsurface
	float subsurface;
	int subsurface_tex;

	//Clearcoat
	float clearcoat;
	int clearcoat_tex;

	//IOR
	float ior;

	//Normalmap
	int normal_tex;

	//BumpMap
	int bump_tex;

	//Emmision
	float3 emmision_color;
	int emmision_color_tex;
	bool is_light;

	//Ideal Specular
	bool ideal_specular = false;
};


std::ostream& operator<<(std::ostream& stream, const Material& f)
{
	stream << "diffuse " << f.base_color << " texID : " << f.base_color_tex << std::endl;
	stream << "specular " << f.specular << " texID : " << f.specular_tex << std::endl;
	stream << "roughness " << f.roughness << " texID : " << f.roughness_tex << std::endl;
	stream << "metallic " << f.metallic << " texID : " << f.metallic_tex << std::endl;
	stream << "sheen " << f.sheen << " texID : " << f.sheen_tex << std::endl;
	stream << "subsurface " << f.subsurface << " texID : " << f.subsurface_tex << std::endl;
	stream << "clearcoat " << f.clearcoat << " texID : " << f.clearcoat_tex << std::endl;
	stream << "ior " << f.ior << " texID : " << f.sheen_tex << std::endl;
	stream << "emission " << f.emmision_color << " texID : " << f.emmision_color_tex << std::endl;

	return stream;
}
