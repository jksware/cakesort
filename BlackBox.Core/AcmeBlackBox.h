#pragma once

#include "BlackBox.h"

namespace BBox {
	namespace Core {

		template <typename T, typename R>
		class AcmeBlackBox final : public BlackBox<T, R>
		{
		public:
			//	AcmeBlackBox() {}
			//	~AcmeBlackBox() {}

			bool Read(R& result) override { return true; }
			bool Process() override { return true; }
			bool Write(const T& input) override { return true; }
			bool Finish() override { return true; }

			std::string Name() override { return "Acme Corporation BlackBox"; }
			std::string ShortName() override { return "Acme Placebo"; }
		};
	}
}
