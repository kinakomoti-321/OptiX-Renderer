#pragma once
#include <iostream>
#include <string>

namespace Log {
	template <typename T>
	inline void DebugLog(const std::string& debugstr, const T& input) {
		std::cout << debugstr << " : " << input << std::endl;
	}

	template <typename T>
	inline void DebugLog(const T& debugstr) {
		std::cout << debugstr << std::endl;
	}

	inline void StartLog(const std::string& str) {
		std::cout << "--------------------------------------" << std::endl;
		std::cout << str << " start" << std::endl;
		std::cout << "--------------------------------------" << std::endl;
	}

	inline void EndLog(const std::string& str) {
		std::cout << "--------------------------------------" << std::endl;
		std::cout << str << " End" << std::endl;
		std::cout << "--------------------------------------" << std::endl;
	}
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec)
{
	for (int i = 0; i < vec.size(); i++) {
		stream << vec[i] << " , ";
	}
	return stream;
}

std::ostream& operator<<(std::ostream& stream, const float3 f)
{
	stream << "(" << f.x << "," << f.y << "," << f.z << ")";
	return stream;
}
