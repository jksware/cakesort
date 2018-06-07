#pragma once
#pragma warning(disable:4018)
#pragma warning(disable:4996)

#include <iostream>

namespace BBox {
	namespace Core {

		template <typename T, typename R>
		class BlackBox
		{
		public:
			BlackBox() {}

			BlackBox(const BlackBox&) = delete;
			BlackBox& operator=(const BlackBox&) = delete;

			// BlackBox(BlackBox&&) = delete;
			// BlackBox& operator=(BlackBox&&) = delete;

			virtual ~BlackBox() {};

			virtual bool Read(R& result) = 0;
			virtual bool Process() = 0;
			virtual bool Write(const T& input) = 0;
			virtual bool Finish() = 0;

			virtual std::string Name() = 0;
			virtual std::string ShortName() = 0;
		};
	}
}
