#pragma once

#include <cuda_runtime.h>

#include "BlackBox.Core/BlackBox.h"
#include "CudaBlackBox.h"

namespace BBox {
	namespace CUDA {
		using namespace BBox::Core;

		template <typename T, typename R>
		class CudaBlackBox : public BlackBox <T, R>
		{
		protected:
			// Holds the length (given the context) of the main data types.
			int _length;

			static const int BlockSize = 128;

		public:
			CudaBlackBox()
			{
			}

			virtual bool Finish() override
			{
				cudaDeviceSynchronize();
				return true;
			}

			inline void length(int length)
			{
				_length = length;
			}

			inline int length() const {
				return _length;
			}

			virtual ~CudaBlackBox() {};
		};
	}
}