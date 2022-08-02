#pragma once
#include <sutil/vec_math.h>
#include <iostream>

struct Affine4x4 {
	float v[16];
	Affine4x4(const float* in_v) {
		for (int i = 0; i < 16; i++) {
			v[i] = in_v[i];
		}
	};
	Affine4x4() {
		for (int i = 0; i < 16; i++) {
			v[i] = 0;
		}
	}
	float operator[](int idx) const{ return v[idx]; }
};

Affine4x4 translateAffine(const float3& t) {
	float v[16] = { 1,0,0,t.x,0,1,0,t.y,0,0,1,t.z,0,0,0,1};
	return Affine4x4(v);
}

Affine4x4 scaleAffine(const float3& s) {
	float v[16] = { s.x,0,0,0, 0,s.y,0,0, 0,0,s.z,0, 0,0,0,1};
	return Affine4x4(v);
}

//Quatanion
Affine4x4 rotateAffine(const float4& q) {
	float q2xy = 2.0 * q.x * q.y;
	float q2xz = 2.0 * q.x * q.z;
	float q2xw = 2.0 * q.x * q.w;
	float q2yz = 2.0 * q.y * q.z;
	float q2yw = 2.0 * q.y * q.w;
	float q2zw = 2.0 * q.z * q.w;
	float q2ww = 2.0 * q.w * q.w;

	float v[16] = {
		q2ww + 2.0f * q.x * q.x - 1.0f, q2xy - q2zw, q2xz + q2yw,0,
		q2xy + q2zw, q2ww + 2.0f * q.y * q.y - 1.0f, q2yz - q2xw,0,
		q2xz - q2yw, q2yz + q2xw, q2ww + 2.0f * q.z * q.z - 1.0f,0,
		0,0,0,1
	};
	/*
	float v[16] = {
		q.x * q.x + q.y * q.y - q.z * q.z - q.w * q.w, 2.0f * (q.y * q.z - q.x * q.w) , 2.0f * (q.x * q.z + q.y * q.w), 0,
		2.0f * (q.x * q.w + q.y * q.z) , q.x * q.x - q.y * q.y + q.z * q.z - q.w * q.w,  2.0f * (-q.x * q.y + q.z * q.w), 0,
		2.0f * (q.y * q.w - q.x * q.z) ,2.0f * (q.z * q.w + q.x * q.y) , q.x * q.x - q.y * q.y - q.z * q.z + q.w * q.w,0,
		0,0,0,1
	};
	*/
	return Affine4x4(v);
}

float4 operator*(const Affine4x4& affine,const float4& p) {
	//Log::DebugLog(p);
	//Log::DebugLog(affine);
	return make_float4(
		p.x * affine[0] + p.y * affine[1] + p.z * affine[2] + p.w * affine[3],
		p.x * affine[4] + p.y * affine[5] + p.z * affine[6] + p.w * affine[7],
		p.x * affine[8] + p.y * affine[9] + p.z * affine[10] + p.w * affine[11],
		p.x * affine[12] + p.y * affine[13] + p.z * affine[14] + p.w * affine[15]
	);
}

Affine4x4 operator*(const Affine4x4& a,const Affine4x4& b) {
	float v[16];
	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			v[i + j * 4] = a[0 + j * 4] * b[i + 0 * 4] + a[1 + j * 4] * b[i + 1 * 4] + a[2 + j * 4] * b[i + 2 * 4] + a[3 + j * 4] * b[i + 3 * 4];
		}
	}

	return Affine4x4(v);
}

std::ostream& operator<<(std::ostream& stream, const Affine4x4& a)
{
	for (int j = 0; j < 4; j++) {
		stream << a[4 * j] << "," << a[1 + 4 * j] << "," << a[2 + 4 * j] << "," << a[3 + 4 * j] << std::endl;
	}
	return stream;
}

float3 quartanionToEuler(const float4& q) {
	float yaw = std::atan(-(2.0 * (q.x * q.y + q.z * q.w))/(q.x * q.x - q.y * q.y - q.z * q.z + q.w * q.w));
	float pitch = std::asin(2.0f * (q.x * q.z - q.y * q.w));
	float roll = std::atan(-2.0f * (q.x * q.w + q.y * q.z) / (q.x * q.x + q.y * q.y - q.z * q.z - q.w * q.w));

	Log::DebugLog("roll", 360.0f * roll / (2.0f * M_PI));
	Log::DebugLog("pitch", 360.0f * pitch / (2.0f * M_PI));
	Log::DebugLog("yaw", 360.0f * yaw / (2.0f * M_PI));
	return { roll,pitch,yaw };
}
