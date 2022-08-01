#pragma once
#include <myPathTracer/matrix.h>
#include <sutil/vec_math.h>
#include <vector>

//ToDo ï‚äÆèàóù
enum AnimationDataType {
	TRANSLATION,
	ROTATION,
	SCALE
};

enum InterpolateType {
	LINER,
	STEP,
	CUBICSPLAINE
};

template<typename T>
struct AnimationData {
	std::vector<T> data;
	std::vector<float> key;
	InterpolateType interpolate_type = LINER;
	AnimationDataType data_type;
	
	AnimationData() {

	}
	AnimationData(const std::vector<T>& data,const std::vector<float>& key,const InterpolateType& interpolate_type,const AnimationDataType& data_type)
		:data(data),key(key),interpolate_type(interpolate_type),data_type(data_type) {}
};

struct Animation{
	std::string animation_name;
	AnimationData<float3> translation_data;
	AnimationData<float4> rotation_data;
	AnimationData<float3> scale_data;

	Animation(){
	}
	template<typename T>	
	T animationInterpolate(const std::vector<T>& animation,const std::vector<float> key, const InterpolateType& type,float time) {
		if (key.size() == 1 || time < 0) return animation[0];
		//ìÒï™íTçıÇ≈offsetÇìæÇÈ
		int first = 0, len = key.size();
		while (len > 0) {
			int half = len >> 1, middle = first + half;
			if (key[middle] <= time) {
				first = middle + 1;
				len -= half + 1;
			}
			else {
				len = half;
			}
		}

		int offset = first - 1;
		if (key.size() >= offset) return animation[key.size() - 1];

		float time_offset = time - key[offset];
		float time_delta = key[offset + 1] - key[offset];
		float delta = time_offset / time_delta;
		
		return interpolate(animation[offset],animation[offset+1],type,delta);
	}
	
	template<typename T>
	T interpolate(const T& a,const T& b, const InterpolateType& type, float delta) {
		switch (type)
		{
		case LINER:
			return a * (1.0f - delta) + b * (delta);
			break;
		default:
			return a * (1.0f - delta) + b * (delta);
			break;
		}
	}

	Affine4x4 getAnimationAffine(float time) {
		float3 translate_frame = (translation_data.key.size() != 0) ? animationInterpolate(translation_data.data, translation_data.key, 
			translation_data.interpolate_type, time): make_float3(0);
		float4 rotate_frame = (rotation_data.key.size() != 0) ? animationInterpolate(rotation_data.data, rotation_data.key, 
			rotation_data.interpolate_type, time): make_float4(0);
		float3 scale_frame = (scale_data.key.size() != 0) ? animationInterpolate(scale_data.data, scale_data.key, 
			scale_data.interpolate_type, time): make_float3(0);
		
		Affine4x4 translate_affine = translateAffine(translate_frame);
		Affine4x4 rotate_affine = rotateAffine(rotate_frame);
		Affine4x4 scale_affine = scaleAffine(scale_frame);

		return translate_affine * rotate_affine * scale_affine;
	}
	
	Affine4x4 getRotateAnimationAffine(float time) {
		float4 rotate_frame = (rotation_data.key.size() != 0) ? animationInterpolate(rotation_data.data, rotation_data.key, 
			rotation_data.interpolate_type, time): make_float4(0);
		Affine4x4 rotate_affine = rotateAffine(rotate_frame);

		return rotate_affine;
	}

	bool DataCheck()const{
		if (translation_data.data.size() != translation_data.key.size()) {
			Log::DebugLog(animation_name);
			Log::DebugLog("TranslationData Error");
			return false;
		}
		if (rotation_data.data.size() != rotation_data.key.size()) {
			Log::DebugLog(animation_name);
			Log::DebugLog("RotationData Error");
			return false;
		}
		if (scale_data.data.size() != scale_data.key.size()) {
			Log::DebugLog(animation_name);
			Log::DebugLog("Scale Error");
			return false;
		}
		Log::DebugLog("Consistency of Animation data is normal");
		return true;
	}
};

std::ostream& operator<<(std::ostream& stream, const Animation& a)
{
	stream << "Translate Data " << std::endl << a.translation_data.data << std::endl;
	stream << "Translate Key "<< std::endl << a.translation_data.key << std::endl;
	stream << "Translate Interpolate "<< std::endl << a.translation_data.interpolate_type << std::endl;

	stream << "Rotation Data "<< std::endl << a.rotation_data.data << std::endl;
	stream << "Rotation Key "<< std::endl << a.rotation_data.key << std::endl;
	stream << "Rotation Interpolate " << std::endl << a.rotation_data.interpolate_type << std::endl;

	stream << "Scale Data " << std::endl<< a.scale_data.data << std::endl;
	stream << "Scale Key " << std::endl<< a.scale_data.key << std::endl;
	stream << "Scale Interpolate "<< std::endl << a.scale_data.interpolate_type << std::endl;
	
	a.DataCheck();

	return stream;
}
