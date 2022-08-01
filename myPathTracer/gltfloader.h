#pragma once

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <external/tinygltf/tiny_gltf.h>

#include <cstdio>
#include <fstream>
#include <iostream>

static std::string GetFilePathExtension(const std::string& FileName) {
	if (FileName.find_last_of(".") != std::string::npos)
		return FileName.substr(FileName.find_last_of(".") + 1);
	return "";
}

static std::string PrintMode(int mode) {
	if (mode == TINYGLTF_MODE_POINTS) {
		return "POINTS";
	}
	else if (mode == TINYGLTF_MODE_LINE) {
		return "LINE";
	}
	else if (mode == TINYGLTF_MODE_LINE_LOOP) {
		return "LINE_LOOP";
	}
	else if (mode == TINYGLTF_MODE_TRIANGLES) {
		return "TRIANGLES";
	}
	else if (mode == TINYGLTF_MODE_TRIANGLE_FAN) {
		return "TRIANGLE_FAN";
	}
	else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
		return "TRIANGLE_STRIP";
	}
	return "**UNKNOWN**";
}

static std::string PrintTarget(int target) {
	if (target == 34962) {
		return "GL_ARRAY_BUFFER";
	}
	else if (target == 34963) {
		return "GL_ELEMENT_ARRAY_BUFFER";
	}
	else {
		return "**UNKNOWN**";
	}
}

static std::string PrintType(int ty) {
	if (ty == TINYGLTF_TYPE_SCALAR) {
		return "SCALAR";
	}
	else if (ty == TINYGLTF_TYPE_VECTOR) {
		return "VECTOR";
	}
	else if (ty == TINYGLTF_TYPE_VEC2) {
		return "VEC2";
	}
	else if (ty == TINYGLTF_TYPE_VEC3) {
		return "VEC3";
	}
	else if (ty == TINYGLTF_TYPE_VEC4) {
		return "VEC4";
	}
	else if (ty == TINYGLTF_TYPE_MATRIX) {
		return "MATRIX";
	}
	else if (ty == TINYGLTF_TYPE_MAT2) {
		return "MAT2";
	}
	else if (ty == TINYGLTF_TYPE_MAT3) {
		return "MAT3";
	}
	else if (ty == TINYGLTF_TYPE_MAT4) {
		return "MAT4";
	}
	return "**UNKNOWN**";
}

static std::string PrintComponentType(int ty) {
	if (ty == TINYGLTF_COMPONENT_TYPE_BYTE) {
		return "BYTE";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
		return "UNSIGNED_BYTE";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_SHORT) {
		return "SHORT";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
		return "UNSIGNED_SHORT";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_INT) {
		return "INT";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
		return "UNSIGNED_INT";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_FLOAT) {
		return "FLOAT";
	}
	else if (ty == TINYGLTF_COMPONENT_TYPE_DOUBLE) {
		return "DOUBLE";
	}

	return "**UNKNOWN**";
}

#if 0
static std::string PrintParameterType(int ty) {
	if (ty == TINYGLTF_PARAMETER_TYPE_BYTE) {
		return "BYTE";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE) {
		return "UNSIGNED_BYTE";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_SHORT) {
		return "SHORT";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT) {
		return "UNSIGNED_SHORT";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_INT) {
		return "INT";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT) {
		return "UNSIGNED_INT";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT) {
		return "FLOAT";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT_VEC2) {
		return "FLOAT_VEC2";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT_VEC3) {
		return "FLOAT_VEC3";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT_VEC4) {
		return "FLOAT_VEC4";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_INT_VEC2) {
		return "INT_VEC2";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_INT_VEC3) {
		return "INT_VEC3";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_INT_VEC4) {
		return "INT_VEC4";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_BOOL) {
		return "BOOL";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_BOOL_VEC2) {
		return "BOOL_VEC2";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_BOOL_VEC3) {
		return "BOOL_VEC3";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_BOOL_VEC4) {
		return "BOOL_VEC4";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT_MAT2) {
		return "FLOAT_MAT2";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT_MAT3) {
		return "FLOAT_MAT3";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_FLOAT_MAT4) {
		return "FLOAT_MAT4";
	}
	else if (ty == TINYGLTF_PARAMETER_TYPE_SAMPLER_2D) {
		return "SAMPLER_2D";
	}

	return "**UNKNOWN**";
}
#endif

static std::string PrintWrapMode(int mode) {
	if (mode == TINYGLTF_TEXTURE_WRAP_REPEAT) {
		return "REPEAT";
	}
	else if (mode == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE) {
		return "CLAMP_TO_EDGE";
	}
	else if (mode == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT) {
		return "MIRRORED_REPEAT";
	}

	return "**UNKNOWN**";
}

static std::string PrintFilterMode(int mode) {
	if (mode == TINYGLTF_TEXTURE_FILTER_NEAREST) {
		return "NEAREST";
	}
	else if (mode == TINYGLTF_TEXTURE_FILTER_LINEAR) {
		return "LINEAR";
	}
	else if (mode == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST) {
		return "NEAREST_MIPMAP_NEAREST";
	}
	else if (mode == TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR) {
		return "NEAREST_MIPMAP_LINEAR";
	}
	else if (mode == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST) {
		return "LINEAR_MIPMAP_NEAREST";
	}
	else if (mode == TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR) {
		return "LINEAR_MIPMAP_LINEAR";
	}
	return "**UNKNOWN**";
}

static std::string PrintIntArray(const std::vector<int>& arr) {
	if (arr.size() == 0) {
		return "";
	}

	std::stringstream ss;
	ss << "[ ";
	for (size_t i = 0; i < arr.size(); i++) {
		ss << arr[i];
		if (i != arr.size() - 1) {
			ss << ", ";
		}
	}
	ss << " ]";

	return ss.str();
}

static std::string PrintFloatArray(const std::vector<double>& arr) {
	if (arr.size() == 0) {
		return "";
	}

	std::stringstream ss;
	ss << "[ ";
	for (size_t i = 0; i < arr.size(); i++) {
		ss << arr[i];
		if (i != arr.size() - 1) {
			ss << ", ";
		}
	}
	ss << " ]";

	return ss.str();
}

static std::string Indent(const int indent) {
	std::string s;
	for (int i = 0; i < indent; i++) {
		s += "  ";
	}

	return s;
}

static std::string PrintParameterValue(const tinygltf::Parameter& param) {
	if (!param.number_array.empty()) {
		return PrintFloatArray(param.number_array);
	}
	else {
		return param.string_value;
	}
}

#if 0
static std::string PrintParameterMap(const tinygltf::ParameterMap& pmap) {
	std::stringstream ss;

	ss << pmap.size() << std::endl;
	for (auto& kv : pmap) {
		ss << kv.first << " : " << PrintParameterValue(kv.second) << std::endl;
	}

	return ss.str();
}
#endif

static std::string PrintValue(const std::string& name,
	const tinygltf::Value& value, const int indent,
	const bool tag = true) {
	std::stringstream ss;

	if (value.IsObject()) {
		const tinygltf::Value::Object& o = value.Get<tinygltf::Value::Object>();
		tinygltf::Value::Object::const_iterator it(o.begin());
		tinygltf::Value::Object::const_iterator itEnd(o.end());
		for (; it != itEnd; it++) {
			ss << PrintValue(it->first, it->second, indent + 1) << std::endl;
		}
	}
	else if (value.IsString()) {
		if (tag) {
			ss << Indent(indent) << name << " : " << value.Get<std::string>();
		}
		else {
			ss << Indent(indent) << value.Get<std::string>() << " ";
		}
	}
	else if (value.IsBool()) {
		if (tag) {
			ss << Indent(indent) << name << " : " << value.Get<bool>();
		}
		else {
			ss << Indent(indent) << value.Get<bool>() << " ";
		}
	}
	else if (value.IsNumber()) {
		if (tag) {
			ss << Indent(indent) << name << " : " << value.Get<double>();
		}
		else {
			ss << Indent(indent) << value.Get<double>() << " ";
		}
	}
	else if (value.IsInt()) {
		if (tag) {
			ss << Indent(indent) << name << " : " << value.Get<int>();
		}
		else {
			ss << Indent(indent) << value.Get<int>() << " ";
		}
	}
	else if (value.IsArray()) {
		// TODO(syoyo): Better pretty printing of array item
		ss << Indent(indent) << name << " [ \n";
		for (size_t i = 0; i < value.Size(); i++) {
			ss << PrintValue("", value.Get(int(i)), indent + 1, /* tag */ false);
			if (i != (value.ArrayLen() - 1)) {
				ss << ", \n";
			}
		}
		ss << "\n" << Indent(indent) << "] ";
	}

	// @todo { binary }

	return ss.str();
}

static void DumpNode(const tinygltf::Node& node, int indent) {
	std::cout << Indent(indent) << "name        : " << node.name << std::endl;
	std::cout << Indent(indent) << "camera      : " << node.camera << std::endl;
	std::cout << Indent(indent) << "mesh        : " << node.mesh << std::endl;
	if (!node.rotation.empty()) {
		std::cout << Indent(indent)
			<< "rotation    : " << PrintFloatArray(node.rotation)
			<< std::endl;
	}
	if (!node.scale.empty()) {
		std::cout << Indent(indent)
			<< "scale       : " << PrintFloatArray(node.scale) << std::endl;
	}
	if (!node.translation.empty()) {
		std::cout << Indent(indent)
			<< "translation : " << PrintFloatArray(node.translation)
			<< std::endl;
	}

	if (!node.matrix.empty()) {
		std::cout << Indent(indent)
			<< "matrix      : " << PrintFloatArray(node.matrix) << std::endl;
	}

	std::cout << Indent(indent)
		<< "children    : " << PrintIntArray(node.children) << std::endl;
}

static void DumpStringIntMap(const std::map<std::string, int>& m, int indent) {
	std::map<std::string, int>::const_iterator it(m.begin());
	std::map<std::string, int>::const_iterator itEnd(m.end());
	for (; it != itEnd; it++) {
		std::cout << Indent(indent) << it->first << ": " << it->second << std::endl;
	}
}

static void DumpExtensions(const tinygltf::ExtensionMap& extension,
	const int indent) {
	// TODO(syoyo): pritty print Value
	for (auto& e : extension) {
		std::cout << Indent(indent) << e.first << std::endl;
		std::cout << PrintValue("extensions", e.second, indent + 1) << std::endl;
	}
}

static void DumpPrimitive(const tinygltf::Primitive& primitive, int indent) {
	std::cout << Indent(indent) << "material : " << primitive.material
		<< std::endl;
	std::cout << Indent(indent) << "indices : " << primitive.indices << std::endl;
	std::cout << Indent(indent) << "mode     : " << PrintMode(primitive.mode)
		<< "(" << primitive.mode << ")" << std::endl;
	std::cout << Indent(indent)
		<< "attributes(items=" << primitive.attributes.size() << ")"
		<< std::endl;
	DumpStringIntMap(primitive.attributes, indent + 1);

	DumpExtensions(primitive.extensions, indent);
	std::cout << Indent(indent) << "extras :" << std::endl
		<< PrintValue("extras", primitive.extras, indent + 1) << std::endl;

	if (!primitive.extensions_json_string.empty()) {
		std::cout << Indent(indent + 1) << "extensions(JSON string) = "
			<< primitive.extensions_json_string << "\n";
	}

	if (!primitive.extras_json_string.empty()) {
		std::cout << Indent(indent + 1)
			<< "extras(JSON string) = " << primitive.extras_json_string
			<< "\n";
	}
}


static void DumpTextureInfo(const tinygltf::TextureInfo& texinfo,
	const int indent) {
	std::cout << Indent(indent) << "index     : " << texinfo.index << "\n";
	std::cout << Indent(indent) << "texCoord  : TEXCOORD_" << texinfo.texCoord
		<< "\n";
	DumpExtensions(texinfo.extensions, indent + 1);
	std::cout << PrintValue("extras", texinfo.extras, indent + 1) << "\n";

	if (!texinfo.extensions_json_string.empty()) {
		std::cout << Indent(indent)
			<< "extensions(JSON string) = " << texinfo.extensions_json_string
			<< "\n";
	}

	if (!texinfo.extras_json_string.empty()) {
		std::cout << Indent(indent)
			<< "extras(JSON string) = " << texinfo.extras_json_string << "\n";
	}
}

static void DumpNormalTextureInfo(const tinygltf::NormalTextureInfo& texinfo,
	const int indent) {
	std::cout << Indent(indent) << "index     : " << texinfo.index << "\n";
	std::cout << Indent(indent) << "texCoord  : TEXCOORD_" << texinfo.texCoord
		<< "\n";
	std::cout << Indent(indent) << "scale     : " << texinfo.scale << "\n";
	DumpExtensions(texinfo.extensions, indent + 1);
	std::cout << PrintValue("extras", texinfo.extras, indent + 1) << "\n";
}

static void DumpOcclusionTextureInfo(
	const tinygltf::OcclusionTextureInfo& texinfo, const int indent) {
	std::cout << Indent(indent) << "index     : " << texinfo.index << "\n";
	std::cout << Indent(indent) << "texCoord  : TEXCOORD_" << texinfo.texCoord
		<< "\n";
	std::cout << Indent(indent) << "strength  : " << texinfo.strength << "\n";
	DumpExtensions(texinfo.extensions, indent + 1);
	std::cout << PrintValue("extras", texinfo.extras, indent + 1) << "\n";
}

static void DumpPbrMetallicRoughness(const tinygltf::PbrMetallicRoughness& pbr,
	const int indent) {
	std::cout << Indent(indent)
		<< "baseColorFactor   : " << PrintFloatArray(pbr.baseColorFactor)
		<< "\n";
	std::cout << Indent(indent) << "baseColorTexture  :\n";
	DumpTextureInfo(pbr.baseColorTexture, indent + 1);

	std::cout << Indent(indent) << "metallicFactor    : " << pbr.metallicFactor
		<< "\n";
	std::cout << Indent(indent) << "roughnessFactor   : " << pbr.roughnessFactor
		<< "\n";

	std::cout << Indent(indent) << "metallicRoughnessTexture  :\n";
	DumpTextureInfo(pbr.metallicRoughnessTexture, indent + 1);
	DumpExtensions(pbr.extensions, indent + 1);
	std::cout << PrintValue("extras", pbr.extras, indent + 1) << "\n";
}

static void Dump(const tinygltf::Model& model) {
	std::cout << "=== Dump glTF ===" << std::endl;
	std::cout << "asset.copyright          : " << model.asset.copyright
		<< std::endl;
	std::cout << "asset.generator          : " << model.asset.generator
		<< std::endl;
	std::cout << "asset.version            : " << model.asset.version
		<< std::endl;
	std::cout << "asset.minVersion         : " << model.asset.minVersion
		<< std::endl;
	std::cout << std::endl;

	std::cout << "=== Dump scene ===" << std::endl;
	std::cout << "defaultScene: " << model.defaultScene << std::endl;

	{
		std::cout << "scenes(items=" << model.scenes.size() << ")" << std::endl;
		for (size_t i = 0; i < model.scenes.size(); i++) {
			std::cout << Indent(1) << "scene[" << i
				<< "] name  : " << model.scenes[i].name << std::endl;
			DumpExtensions(model.scenes[i].extensions, 1);
		}
	}

	{
		std::cout << "meshes(item=" << model.meshes.size() << ")" << std::endl;
		for (size_t i = 0; i < model.meshes.size(); i++) {
			std::cout << Indent(1) << "name     : " << model.meshes[i].name
				<< std::endl;
			std::cout << Indent(1)
				<< "primitives(items=" << model.meshes[i].primitives.size()
				<< "): " << std::endl;

			for (size_t k = 0; k < model.meshes[i].primitives.size(); k++) {
				DumpPrimitive(model.meshes[i].primitives[k], 2);
			}
		}
	}

	{
		for (size_t i = 0; i < model.accessors.size(); i++) {
			const tinygltf::Accessor& accessor = model.accessors[i];
			std::cout << Indent(1) << "name         : " << accessor.name << std::endl;
			std::cout << Indent(2) << "bufferView   : " << accessor.bufferView
				<< std::endl;
			std::cout << Indent(2) << "byteOffset   : " << accessor.byteOffset
				<< std::endl;
			std::cout << Indent(2) << "componentType: "
				<< PrintComponentType(accessor.componentType) << "("
				<< accessor.componentType << ")" << std::endl;
			std::cout << Indent(2) << "count        : " << accessor.count
				<< std::endl;
			std::cout << Indent(2) << "type         : " << PrintType(accessor.type)
				<< std::endl;
			if (!accessor.minValues.empty()) {
				std::cout << Indent(2) << "min          : [";
				for (size_t k = 0; k < accessor.minValues.size(); k++) {
					std::cout << accessor.minValues[k]
						<< ((k != accessor.minValues.size() - 1) ? ", " : "");
				}
				std::cout << "]" << std::endl;
			}
			if (!accessor.maxValues.empty()) {
				std::cout << Indent(2) << "max          : [";
				for (size_t k = 0; k < accessor.maxValues.size(); k++) {
					std::cout << accessor.maxValues[k]
						<< ((k != accessor.maxValues.size() - 1) ? ", " : "");
				}
				std::cout << "]" << std::endl;
			}

			if (accessor.sparse.isSparse) {
				std::cout << Indent(2) << "sparse:" << std::endl;
				std::cout << Indent(3) << "count  : " << accessor.sparse.count
					<< std::endl;
				std::cout << Indent(3) << "indices: " << std::endl;
				std::cout << Indent(4)
					<< "bufferView   : " << accessor.sparse.indices.bufferView
					<< std::endl;
				std::cout << Indent(4)
					<< "byteOffset   : " << accessor.sparse.indices.byteOffset
					<< std::endl;
				std::cout << Indent(4) << "componentType: "
					<< PrintComponentType(accessor.sparse.indices.componentType)
					<< "(" << accessor.sparse.indices.componentType << ")"
					<< std::endl;
				std::cout << Indent(3) << "values : " << std::endl;
				std::cout << Indent(4)
					<< "bufferView   : " << accessor.sparse.values.bufferView
					<< std::endl;
				std::cout << Indent(4)
					<< "byteOffset   : " << accessor.sparse.values.byteOffset
					<< std::endl;
			}
		}
	}

	{
		std::cout << "animations(items=" << model.animations.size() << ")"
			<< std::endl;
		for (size_t i = 0; i < model.animations.size(); i++) {
			const tinygltf::Animation& animation = model.animations[i];
			std::cout << Indent(1) << "name         : " << animation.name
				<< std::endl;

			std::cout << Indent(1) << "channels : [ " << std::endl;
			for (size_t j = 0; j < animation.channels.size(); j++) {
				std::cout << Indent(2)
					<< "sampler     : " << animation.channels[j].sampler
					<< std::endl;
				std::cout << Indent(2)
					<< "target.id   : " << animation.channels[j].target_node
					<< std::endl;
				std::cout << Indent(2)
					<< "target.path : " << animation.channels[j].target_path
					<< std::endl;
				std::cout << ((i != (animation.channels.size() - 1)) ? "  , " : "");
			}
			std::cout << "  ]" << std::endl;

			std::cout << Indent(1) << "samplers(items=" << animation.samplers.size()
				<< ")" << std::endl;
			for (size_t j = 0; j < animation.samplers.size(); j++) {
				const tinygltf::AnimationSampler& sampler = animation.samplers[j];
				std::cout << Indent(2) << "input         : " << sampler.input
					<< std::endl;
				std::cout << Indent(2) << "interpolation : " << sampler.interpolation
					<< std::endl;
				std::cout << Indent(2) << "output        : " << sampler.output
					<< std::endl;
			}
		}
	}

	{
		std::cout << "bufferViews(items=" << model.bufferViews.size() << ")"
			<< std::endl;
		for (size_t i = 0; i < model.bufferViews.size(); i++) {
			const tinygltf::BufferView& bufferView = model.bufferViews[i];
			std::cout << Indent(1) << "name         : " << bufferView.name
				<< std::endl;
			std::cout << Indent(2) << "buffer       : " << bufferView.buffer
				<< std::endl;
			std::cout << Indent(2) << "byteLength   : " << bufferView.byteLength
				<< std::endl;
			std::cout << Indent(2) << "byteOffset   : " << bufferView.byteOffset
				<< std::endl;
			std::cout << Indent(2) << "byteStride   : " << bufferView.byteStride
				<< std::endl;
			std::cout << Indent(2)
				<< "target       : " << PrintTarget(bufferView.target)
				<< std::endl;
			std::cout << Indent(1) << "-------------------------------------\n";

			DumpExtensions(bufferView.extensions, 1);
			std::cout << PrintValue("extras", bufferView.extras, 2) << std::endl;

			if (!bufferView.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< bufferView.extensions_json_string << "\n";
			}

			if (!bufferView.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << bufferView.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "buffers(items=" << model.buffers.size() << ")" << std::endl;
		for (size_t i = 0; i < model.buffers.size(); i++) {
			const tinygltf::Buffer& buffer = model.buffers[i];
			std::cout << Indent(1) << "name         : " << buffer.name << std::endl;
			std::cout << Indent(2) << "byteLength   : " << buffer.data.size()
				<< std::endl;
			std::cout << Indent(1) << "-------------------------------------\n";

			DumpExtensions(buffer.extensions, 1);
			std::cout << PrintValue("extras", buffer.extras, 2) << std::endl;

			if (!buffer.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< buffer.extensions_json_string << "\n";
			}

			if (!buffer.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << buffer.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "materials(items=" << model.materials.size() << ")"
			<< std::endl;
		for (size_t i = 0; i < model.materials.size(); i++) {
			const tinygltf::Material& material = model.materials[i];
			std::cout << Indent(1) << "name                 : " << material.name
				<< std::endl;

			std::cout << Indent(1) << "alphaMode            : " << material.alphaMode
				<< std::endl;
			std::cout << Indent(1)
				<< "alphaCutoff          : " << material.alphaCutoff
				<< std::endl;
			std::cout << Indent(1) << "doubleSided          : "
				<< (material.doubleSided ? "true" : "false") << std::endl;
			std::cout << Indent(1) << "emissiveFactor       : "
				<< PrintFloatArray(material.emissiveFactor) << std::endl;

			std::cout << Indent(1) << "pbrMetallicRoughness :\n";
			DumpPbrMetallicRoughness(material.pbrMetallicRoughness, 2);

			std::cout << Indent(1) << "normalTexture        :\n";
			DumpNormalTextureInfo(material.normalTexture, 2);

			std::cout << Indent(1) << "occlusionTexture     :\n";
			DumpOcclusionTextureInfo(material.occlusionTexture, 2);

			std::cout << Indent(1) << "emissiveTexture      :\n";
			DumpTextureInfo(material.emissiveTexture, 2);

			std::cout << Indent(1) << "----  legacy material parameter  ----\n";
			std::cout << Indent(1) << "values(items=" << material.values.size() << ")"
				<< std::endl;
			tinygltf::ParameterMap::const_iterator p(material.values.begin());
			tinygltf::ParameterMap::const_iterator pEnd(material.values.end());
			for (; p != pEnd; p++) {
				std::cout << Indent(2) << p->first << ": "
					<< PrintParameterValue(p->second) << std::endl;
			}
			std::cout << Indent(1) << "-------------------------------------\n";

			DumpExtensions(material.extensions, 1);
			std::cout << PrintValue("extras", material.extras, 2) << std::endl;

			if (!material.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< material.extensions_json_string << "\n";
			}

			if (!material.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << material.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "nodes(items=" << model.nodes.size() << ")" << std::endl;
		for (size_t i = 0; i < model.nodes.size(); i++) {
			const tinygltf::Node& node = model.nodes[i];
			std::cout << Indent(1) << "name         : " << node.name << std::endl;

			DumpNode(node, 2);
		}
	}

	{
		std::cout << "images(items=" << model.images.size() << ")" << std::endl;
		for (size_t i = 0; i < model.images.size(); i++) {
			const tinygltf::Image& image = model.images[i];
			std::cout << Indent(1) << "name         : " << image.name << std::endl;

			std::cout << Indent(2) << "width     : " << image.width << std::endl;
			std::cout << Indent(2) << "height    : " << image.height << std::endl;
			std::cout << Indent(2) << "component : " << image.component << std::endl;
			DumpExtensions(image.extensions, 1);
			std::cout << PrintValue("extras", image.extras, 2) << std::endl;

			if (!image.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< image.extensions_json_string << "\n";
			}

			if (!image.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << image.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "textures(items=" << model.textures.size() << ")" << std::endl;
		for (size_t i = 0; i < model.textures.size(); i++) {
			const tinygltf::Texture& texture = model.textures[i];
			std::cout << Indent(1) << "sampler        : " << texture.sampler
				<< std::endl;
			std::cout << Indent(1) << "source         : " << texture.source
				<< std::endl;
			DumpExtensions(texture.extensions, 1);
			std::cout << PrintValue("extras", texture.extras, 2) << std::endl;

			if (!texture.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< texture.extensions_json_string << "\n";
			}

			if (!texture.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << texture.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "samplers(items=" << model.samplers.size() << ")" << std::endl;

		for (size_t i = 0; i < model.samplers.size(); i++) {
			const tinygltf::Sampler& sampler = model.samplers[i];
			std::cout << Indent(1) << "name (id)    : " << sampler.name << std::endl;
			std::cout << Indent(2)
				<< "minFilter    : " << PrintFilterMode(sampler.minFilter)
				<< std::endl;
			std::cout << Indent(2)
				<< "magFilter    : " << PrintFilterMode(sampler.magFilter)
				<< std::endl;
			std::cout << Indent(2)
				<< "wrapS        : " << PrintWrapMode(sampler.wrapS)
				<< std::endl;
			std::cout << Indent(2)
				<< "wrapT        : " << PrintWrapMode(sampler.wrapT)
				<< std::endl;

			DumpExtensions(sampler.extensions, 1);
			std::cout << PrintValue("extras", sampler.extras, 2) << std::endl;

			if (!sampler.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< sampler.extensions_json_string << "\n";
			}

			if (!sampler.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << sampler.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "cameras(items=" << model.cameras.size() << ")" << std::endl;

		for (size_t i = 0; i < model.cameras.size(); i++) {
			const tinygltf::Camera& camera = model.cameras[i];
			std::cout << Indent(1) << "name (id)    : " << camera.name << std::endl;
			std::cout << Indent(1) << "type         : " << camera.type << std::endl;

			if (camera.type.compare("perspective") == 0) {
				std::cout << Indent(2)
					<< "aspectRatio   : " << camera.perspective.aspectRatio
					<< std::endl;
				std::cout << Indent(2) << "yfov          : " << camera.perspective.yfov
					<< std::endl;
				std::cout << Indent(2) << "zfar          : " << camera.perspective.zfar
					<< std::endl;
				std::cout << Indent(2) << "znear         : " << camera.perspective.znear
					<< std::endl;
			}
			else if (camera.type.compare("orthographic") == 0) {
				std::cout << Indent(2) << "xmag          : " << camera.orthographic.xmag
					<< std::endl;
				std::cout << Indent(2) << "ymag          : " << camera.orthographic.ymag
					<< std::endl;
				std::cout << Indent(2) << "zfar          : " << camera.orthographic.zfar
					<< std::endl;
				std::cout << Indent(2)
					<< "znear         : " << camera.orthographic.znear
					<< std::endl;
			}

			std::cout << Indent(1) << "-------------------------------------\n";

			DumpExtensions(camera.extensions, 1);
			std::cout << PrintValue("extras", camera.extras, 2) << std::endl;

			if (!camera.extensions_json_string.empty()) {
				std::cout << Indent(2) << "extensions(JSON string) = "
					<< camera.extensions_json_string << "\n";
			}

			if (!camera.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << camera.extras_json_string
					<< "\n";
			}
		}
	}

	{
		std::cout << "skins(items=" << model.skins.size() << ")" << std::endl;
		for (size_t i = 0; i < model.skins.size(); i++) {
			const tinygltf::Skin& skin = model.skins[i];
			std::cout << Indent(1) << "name         : " << skin.name << std::endl;
			std::cout << Indent(2)
				<< "inverseBindMatrices   : " << skin.inverseBindMatrices
				<< std::endl;
			std::cout << Indent(2) << "skeleton              : " << skin.skeleton
				<< std::endl;
			std::cout << Indent(2)
				<< "joints                : " << PrintIntArray(skin.joints)
				<< std::endl;
			std::cout << Indent(1) << "-------------------------------------\n";

			DumpExtensions(skin.extensions, 1);
			std::cout << PrintValue("extras", skin.extras, 2) << std::endl;

			if (!skin.extensions_json_string.empty()) {
				std::cout << Indent(2)
					<< "extensions(JSON string) = " << skin.extensions_json_string
					<< "\n";
			}

			if (!skin.extras_json_string.empty()) {
				std::cout << Indent(2)
					<< "extras(JSON string) = " << skin.extras_json_string
					<< "\n";
			}
		}
	}

	// toplevel extensions
	{
		std::cout << "extensions(items=" << model.extensions.size() << ")"
			<< std::endl;
		DumpExtensions(model.extensions, 1);
	}
}

template <typename T>
struct arrayAdapter {
	/// Pointer to the bytes
	const unsigned char* dataPtr;
	/// Number of elements in the array
	const size_t elemCount;
	/// Stride in bytes between two elements
	const size_t stride;

	/// Construct an array adapter.
	/// \param ptr Pointer to the start of the data, with offset applied
	/// \param count Number of elements in the array
	/// \param byte_stride Stride betweens elements in the array
	arrayAdapter(const unsigned char* ptr, size_t count, size_t byte_stride)
		: dataPtr(ptr), elemCount(count), stride(byte_stride) {}

	/// Returns a *copy* of a single element. Can't be used to modify it.
	T operator[](size_t pos) const {
		if (pos >= elemCount)
			throw std::out_of_range(
				"Tried to access beyond the last element of an array adapter with "
				"count " +
				std::to_string(elemCount) + " while getting elemnet number " +
				std::to_string(pos));
		return *(reinterpret_cast<const T*>(dataPtr + pos * stride));
	}
};

template <typename T>
struct v2 {
	T x, y;
};
/// 3D vector of floats without padding
template <typename T>
struct v3 {
	T x, y, z;
};

/// 4D vector of floats without padding
template <typename T>
struct v4 {
	T x, y, z, w;
};

#pragma pack(pop)

using v2f = v2<float>;
using v3f = v3<float>;
using v4f = v4<float>;
using v2d = v2<double>;
using v3d = v3<double>;
using v4d = v4<double>;

float3 v3f_convert_float3(const v3f& v) {
	return make_float3(v.x, v.y, v.z);
}

float2 v2f_convert_float2(const v2f& v) {
	return make_float2(v.x, v.y);
}

struct intArrayBase {
	virtual ~intArrayBase() = default;
	virtual unsigned int operator[](size_t) const = 0;
	virtual size_t size() const = 0;
};

template <class T>
struct intArray : public intArrayBase {
	arrayAdapter<T> adapter;

	intArray(const arrayAdapter<T>& a) : adapter(a) {}
	unsigned int operator[](size_t position) const override {
		return static_cast<unsigned int>(adapter[position]);
	}

	size_t size() const override { return adapter.elemCount; }
};

struct unsignedIntArray {
	arrayAdapter<unsigned int> adapter;
	unsignedIntArray(const arrayAdapter<unsigned int>& a) : adapter(a) {}

	int operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};

struct floatArray {
	arrayAdapter<float> adapter;
	floatArray(const arrayAdapter<float>& a) : adapter(a) {}

	float operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};
struct v2fArray {
	arrayAdapter<v2f> adapter;
	v2fArray(const arrayAdapter<v2f>& a) : adapter(a) {}

	v2f operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};

struct v3fArray {
	arrayAdapter<v3f> adapter;
	v3fArray(const arrayAdapter<v3f>& a) : adapter(a) {}

	v3f operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};

struct v4fArray {
	arrayAdapter<v4f> adapter;
	v4fArray(const arrayAdapter<v4f>& a) : adapter(a) {}

	v4f operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};

struct v2dArray {
	arrayAdapter<v2d> adapter;
	v2dArray(const arrayAdapter<v2d>& a) : adapter(a) {}

	v2d operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};

struct v3dArray {
	arrayAdapter<v3d> adapter;
	v3dArray(const arrayAdapter<v3d>& a) : adapter(a) {}

	v3d operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};

struct v4dArray {
	arrayAdapter<v4d> adapter;
	v4dArray(const arrayAdapter<v4d>& a) : adapter(a) {}

	v4d operator[](size_t position) const { return adapter[position]; }
	size_t size() const { return adapter.elemCount; }
};


bool gltfloader(std::string& filepath, SceneData& scenedata) {

	// Store original JSON string for `extras` and `extensions`
	bool store_original_json_for_extras_and_extensions = true;

	tinygltf::Model model;
	tinygltf::TinyGLTF gltf_ctx;
	std::string err;
	std::string warn;
	std::string input_filename = filepath;
	std::string ext = GetFilePathExtension(input_filename);

	gltf_ctx.SetStoreOriginalJSONForExtrasAndExtensions(
		store_original_json_for_extras_and_extensions);

	bool ret = false;
	if (ext.compare("glb") == 0) {
		std::cout << "Reading binary glTF" << std::endl;
		// assume binary glTF.
		ret = gltf_ctx.LoadBinaryFromFile(&model, &err, &warn,
			input_filename.c_str());
	}
	else {
		std::cout << "Reading ASCII glTF" << std::endl;
		// assume ascii glTF.
		ret =
			gltf_ctx.LoadASCIIFromFile(&model, &err, &warn, input_filename.c_str());
	}

	if (!warn.empty()) {
		printf("Warn: %s\n", warn.c_str());
	}

	if (!err.empty()) {
		printf("Err: %s\n", err.c_str());
	}

	if (!ret) {
		printf("Failed to parse glTF\n");
		return false;
	}

	//Dump(model);
	//Texture
	for (auto& image : model.images) {
		Log::DebugLog(image.uri);
	}

	std::vector<Animation> animation;
	animation.resize(model.nodes.size());
	//Material
	for (int i = 0; i < model.materials.size(); i++) {
		auto material = model.materials[i];
		auto mat_pram = material.pbrMetallicRoughness;

		Material mat;
		mat.material_name = material.name;

		mat.base_color = { float(mat_pram.baseColorFactor[0]),float(mat_pram.baseColorFactor[1]),float(mat_pram.baseColorFactor[2]) };
		mat.base_color_tex = mat_pram.baseColorTexture.index;

		mat.roughness = float(mat_pram.roughnessFactor);

		mat.roughness_tex = mat_pram.metallicRoughnessTexture.index;

		mat.metallic = float(mat_pram.metallicFactor);
		mat.metallic_tex = mat_pram.metallicRoughnessTexture.index;

		mat.emmision_color = { float(material.emissiveFactor[0]),float(material.emissiveFactor[1]),float(material.emissiveFactor[2]) };
		mat.emmision_color_tex = material.emissiveTexture.index;

		mat.normal_tex = material.normalTexture.index;

		mat.sheen = 0;
		mat.sheen_tex = -1;

		mat.clearcoat = 0;
		mat.clearcoat_tex = -1;

		mat.subsurface = 0;
		mat.subsurface_tex = -1;

		mat.specular = { 0,0,0 };
		mat.specular_tex = -1;

		mat.bump_tex = -1;
		mat.ior = 1.0;

		for (auto& mat_extensions : material.extensions) {
			Log::DebugLog(mat_extensions.first);
			if (mat_extensions.first == "KHR_materials_clearcoat") {
				const tinygltf::Value::Object& o = mat_extensions.second.Get<tinygltf::Value::Object>();
				tinygltf::Value::Object::const_iterator it(o.begin());
				tinygltf::Value::Object::const_iterator itEnd(o.end());
				for (; it != itEnd; it++) {
					if (it->first == "clearcoatRoughnessFactor") {
						mat.clearcoat = it->second.Get<double>();
					}
				}
			}
			else if (mat_extensions.first == "KHR_materials_sheen") {
				const tinygltf::Value::Object& o = mat_extensions.second.Get<tinygltf::Value::Object>();
				tinygltf::Value::Object::const_iterator it(o.begin());
				tinygltf::Value::Object::const_iterator itEnd(o.end());
				for (; it != itEnd; it++) {

				}
			}
			else if (mat_extensions.first == "KHR_materials_transmission") {
				const tinygltf::Value::Object& o = mat_extensions.second.Get<tinygltf::Value::Object>();
				tinygltf::Value::Object::const_iterator it(o.begin());
				tinygltf::Value::Object::const_iterator itEnd(o.end());
				for (; it != itEnd; it++) {

				}
			}
			else if (mat_extensions.first == "KHR_materials_ior") {
				const tinygltf::Value::Object& o = mat_extensions.second.Get<tinygltf::Value::Object>();
				tinygltf::Value::Object::const_iterator it(o.begin());
				tinygltf::Value::Object::const_iterator itEnd(o.end());
				for (; it != itEnd; it++) {

				}
			}
		}

		mat.ideal_specular = false;
		scenedata.material.push_back(mat);
		Log::DebugLog(mat);
	}

	//GeometryData
	{
		int node_index = 0;
		bool cameraCheck = false;
		for (auto& nodes : model.nodes) {
			Log::DebugLog(nodes.camera);
			//AnimationÇÃèâä˙ê›íË
			{
				auto& node_animation = animation[node_index];
				node_animation.translation_data.key.push_back(0);
				node_animation.rotation_data.key.push_back(0);
				node_animation.scale_data.key.push_back(0);

				auto node_translation = nodes.translation;
				auto node_rotation = nodes.rotation;
				auto node_scale = nodes.scale;

				if (node_translation.size() != 0) {
					node_animation.translation_data.data.push_back(make_float3(node_translation[0], node_translation[1], node_translation[2]));
				}
				else {
					node_animation.translation_data.data.push_back(make_float3(0));
				}

				if (node_rotation.size() != 0) {
					node_animation.rotation_data.data.push_back(make_float4(node_rotation[0], node_rotation[1], node_rotation[2], node_rotation[3]));
				}
				else {
					node_animation.rotation_data.data.push_back(make_float4(0, 0, 0, 1));
				}

				if (node_scale.size() != 0) {
					node_animation.scale_data.data.push_back(make_float3(node_scale[0], node_scale[1], node_scale[2]));
				}
				else {
					node_animation.scale_data.data.push_back(make_float3(1));
				}
			}

			//Mesh
			if (nodes.camera == -1) {
				auto meshs = model.meshes[nodes.mesh];
				GASData gasdata;
				gasdata.vert_offset = scenedata.vertices.size();
				unsigned int mesh_poly = 0;
				for (auto& primitives : meshs.primitives) {
					std::unique_ptr<intArrayBase> indicesArrayPtr;
					{
						auto indicesAccessor = model.accessors[primitives.indices];
						auto& indexBufferView = model.bufferViews[indicesAccessor.bufferView];
						auto& indexBuffer = model.buffers[indexBufferView.buffer];
						auto indexPtr = indexBuffer.data.data() + indexBufferView.byteOffset + indicesAccessor.byteOffset;
						auto indexByte_stride = indicesAccessor.ByteStride(indexBufferView);
						auto index_count = indicesAccessor.count;

						switch (indicesAccessor.componentType) {
						case TINYGLTF_COMPONENT_TYPE_BYTE:
							indicesArrayPtr =
								std::unique_ptr<intArray<char> >(new intArray<char>(
									arrayAdapter<char>(indexPtr, index_count, indexByte_stride)));
							break;

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
							indicesArrayPtr = std::unique_ptr<intArray<unsigned char> >(
								new intArray<unsigned char>(arrayAdapter<unsigned char>(
									indexPtr, index_count, indexByte_stride)));
							break;

						case TINYGLTF_COMPONENT_TYPE_SHORT:
							indicesArrayPtr =
								std::unique_ptr<intArray<short> >(new intArray<short>(
									arrayAdapter<short>(indexPtr, index_count, indexByte_stride)));
							break;

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
							indicesArrayPtr = std::unique_ptr<intArray<unsigned short> >(
								new intArray<unsigned short>(arrayAdapter<unsigned short>(
									indexPtr, index_count, indexByte_stride)));
							break;

						case TINYGLTF_COMPONENT_TYPE_INT:
							indicesArrayPtr = std::unique_ptr<intArray<int> >(new intArray<int>(
								arrayAdapter<int>(indexPtr, index_count, indexByte_stride)));
							break;

						case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
							indicesArrayPtr = std::unique_ptr<intArray<unsigned int> >(
								new intArray<unsigned int>(arrayAdapter<unsigned int>(
									indexPtr, index_count, indexByte_stride)));
							break;
						default:
							break;
						}

					}

					const auto& indices = *indicesArrayPtr;

					//Attribute
					//Position,Normal,Texcoord

					std::unique_ptr<v3fArray> vertices = nullptr;
					std::unique_ptr<v3fArray> normals = nullptr;
					std::unique_ptr<v2fArray> texcoords = nullptr;

					{
						for (auto& attribute : primitives.attributes) {

							auto attribAccessor = model.accessors[attribute.second];
							auto& bufferView = model.bufferViews[attribAccessor.bufferView];
							auto& buffer = model.buffers[bufferView.buffer];
							auto dataPtr = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
							auto byte_stride = attribAccessor.ByteStride(bufferView);
							auto count = attribAccessor.count;

							//Log::DebugLog(attribute.first);

							if (attribute.first == "POSITION") {
								vertices = std::make_unique<v3fArray>(arrayAdapter<v3f>(dataPtr, count, byte_stride));
							}
							else if (attribute.first == "NORMAL") {
								normals = std::make_unique<v3fArray>(arrayAdapter<v3f>(dataPtr, count, byte_stride));
							}
							else if (attribute.first == "TEXCOORD_0") {
								texcoords = std::make_unique<v2fArray>(arrayAdapter<v2f>(dataPtr, count, byte_stride));
							}
						}
					}


					auto& vertices_ptr = *vertices;
					auto& normals_ptr = *normals;
					auto& texcoords_ptr = *texcoords;
					for (int i = 0; i < indices.size() / 3; i++) {
						unsigned int idx[3];
						idx[0] = indices[i * 3];
						idx[1] = indices[i * 3 + 1];
						idx[2] = indices[i * 3 + 2];

						float3 vert[3];
						vert[0] = v3f_convert_float3(vertices_ptr[idx[0]]);
						vert[1] = v3f_convert_float3(vertices_ptr[idx[1]]);
						vert[2] = v3f_convert_float3(vertices_ptr[idx[2]]);

						float3 norm[3];
						if (normals != nullptr) {
							norm[0] = v3f_convert_float3(normals_ptr[idx[0]]);
							norm[1] = v3f_convert_float3(normals_ptr[idx[1]]);
							norm[2] = v3f_convert_float3(normals_ptr[idx[2]]);
						}
						else {
							float3 geo_normal = normalize(cross(vert[1] - vert[0], vert[2] - vert[0]));
							norm[0] = geo_normal;
							norm[1] = geo_normal;
							norm[2] = geo_normal;
						}

						float2 texc[3];
						if (texcoords != nullptr) {
							texc[0] = v2f_convert_float2(texcoords_ptr[idx[0]]);
							texc[1] = v2f_convert_float2(texcoords_ptr[idx[1]]);
							texc[2] = v2f_convert_float2(texcoords_ptr[idx[2]]);
						}
						else {
							texc[0] = { 0,0 };
							texc[1] = { 0,0 };
							texc[2] = { 0,0 };
						}

						for (int j = 0; j < 3; j++) {
							//Log::DebugLog(vert[j]);
							//Log::DebugLog(norm[j]);
							//Log::DebugLog(texc[j]);
							scenedata.vertices.push_back(vert[j]);
							scenedata.normal.push_back(norm[j]);
							scenedata.uv.push_back(texc[j]);
						}
						//Log::DebugLog("material", primitives.material);
						scenedata.material_index.push_back(primitives.material);

					}
					mesh_poly += indices.size() / 3;
				}
				gasdata.poly_n = mesh_poly;
			}
			else {
				//Camera
				scenedata.camera.origin = { 0,0,0 };
				scenedata.camera.direciton = { 0,1,0 };
				scenedata.camera.f = 2;
				cameraCheck = true;
				scenedata.CameraAnimationIndex = node_index;
			}

			node_index += 1;
		}
		if (!cameraCheck) {
			scenedata.camera.origin = { 0,0,0 };
			scenedata.camera.direciton = { 0,1,0 };
			scenedata.camera.f = 2;
			scenedata.CameraAnimationIndex = -1;
		}
	}
	

	//Animation
	{
		//(node_index,deta)
		for (auto& anim : model.animations) {
			for (int i = 0; i < anim.channels.size(); i++) {
				auto sampler = anim.samplers[i];
				auto channel = anim.channels[i];
				auto animKeyAccessor = model.accessors[sampler.input];
				auto animDataAccessor = model.accessors[sampler.output];
				Log::DebugLog(channel.target_node);
				auto node = model.nodes[channel.target_node];

				auto& keyBufferView = model.bufferViews[animKeyAccessor.bufferView];
				auto& keyBuffer = model.buffers[keyBufferView.buffer];
				auto keyPtr = keyBuffer.data.data() + keyBufferView.byteOffset + animKeyAccessor.byteOffset;
				auto keyByte_stride = animKeyAccessor.ByteStride(keyBufferView);
				auto key_count = animKeyAccessor.count;

				floatArray keyadapter(arrayAdapter<float>(keyPtr, key_count, keyByte_stride));

				auto& animDataBufferView = model.bufferViews[animDataAccessor.bufferView];
				auto& animDataBuffer = model.buffers[animDataBufferView.buffer];
				auto animDataPtr = animDataBuffer.data.data() + animDataBufferView.byteOffset + animDataAccessor.byteOffset;
				auto animDataByte_stride = animDataAccessor.ByteStride(animDataBufferView);
				auto animData_count = animDataAccessor.count;
				if (channel.target_path == "translation") {
					v3fArray translation_in(arrayAdapter<v3f>(animDataPtr, animData_count, animDataByte_stride));

					for (int i = 0; i < translation_in.size(); i++) {
						animation[channel.target_node].translation_data.data.push_back(make_float3(translation_in[i].x, translation_in[i].y, translation_in[i].z));
						animation[channel.target_node].translation_data.key.push_back(keyadapter[i]);
					}
				}
				else if (channel.target_path == "rotation") {
					v4fArray rotation_in(arrayAdapter<v4f>(animDataPtr, animData_count, animDataByte_stride));

					for (int i = 0; i < rotation_in.size(); i++) {
						animation[channel.target_node].rotation_data.data.push_back(make_float4(rotation_in[i].x, rotation_in[i].y, rotation_in[i].z, rotation_in[i].w));
						animation[channel.target_node].rotation_data.key.push_back(keyadapter[i]);
					}

				}
				else if (channel.target_path == "scale") {
					v3fArray scale_in(arrayAdapter<v3f>(animDataPtr, animData_count, animDataByte_stride));

					for (int i = 0; i < scale_in.size(); i++) {
						animation[channel.target_node].scale_data.data.push_back(make_float3(scale_in[i].x, scale_in[i].y, scale_in[i].z));
						animation[channel.target_node].scale_data.key.push_back(keyadapter[i]);
					}
				}

			}

		}
		/*
		for (auto& anim : animation) {
			Log::DebugLog(anim);
		}
		*/
	}
	scenedata.animation = animation;
	return true;
}
