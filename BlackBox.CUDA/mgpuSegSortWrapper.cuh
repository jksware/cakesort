#pragma once

#include "CudaBlackBox.h"
#include "BlackBox.Core/SegmentedArray.h"

#include <moderngpu/kernel_segsort.hxx>

#include <string>

namespace BBox {
	namespace CUDA {
		using namespace BBox::Core;

		class mgpuSegSortWrapper : public CudaBlackBox<SegmentedArray<int>, SegmentedArray<int>>
		{
			//typedef T myType;
			//typedef CudaBlackBox<SegmentedArray<T>, std::vector<T>> myBase;

			int segmentCount;
			int segmentActiveLength;
			
			mgpu::standard_context_t context;
			mgpu::mem_t<int> segs;
      		mgpu::mem_t<int> data;
			std::vector<int> _segments;

		public:
			std::string Name() override { return "ModernGPU Segmented Sort Wrapper"; }
			std::string ShortName() override { return "mgpuSegSort"; }

			bool Process() override;

			bool Write(const SegmentedArray<int>& input) override;

			bool Read(SegmentedArray<int>& result) override;
		};
	}
}
