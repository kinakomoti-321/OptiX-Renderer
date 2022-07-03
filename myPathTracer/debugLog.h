#pragma once
#include <iostream>
#include <string>

namespace Log {
	template <typename T>
	inline void DebugLog(const std::string& debugstr, const T& input) {
		std::cout << debugstr << " : " << input << std::endl;
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
