#pragma once

#include "OpenCLBlackBox.h"
#include "BlackBox.Core/SegmentedArray.h"
#include "BlackBox.Core/ExtendedStd.h"

#include <string>
#include <exception>

namespace BBox {
	namespace OpenCL {
		using namespace BBox::Core;

		template <typename T>
		class IteratedDispatch : public OpenCLBlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			typedef T myType;
			typedef OpenCLBlackBox<SegmentedArray<T>, SegmentedArray<T>> myBase;

			const std::string KernelPath = std::string("Kernels") + SEP + "BitonicSort_Kernels.cl";
			const std::string KernelName = "bitonicSort";

			cl::Kernel _bitonicSortKernel;
			cl::Buffer _buffer;
			size_t kernelWorkGroupSize;

			std::vector<int> segmentLengths;

			using myBase::_length;

		public:
			IteratedDispatch() = delete;

			IteratedDispatch(cl::Context context, cl::Device device, cl::CommandQueue queue)
				: myBase{context, device, queue }
			{
				if (std::is_scalar<myType>::value) {
					this->declareTypes["__T__"] = typeid(myType).hash_code();
				}
				else {
					// todo don't know what to do, maybe throw an exception
					throw std::runtime_error("Cannot create on non-scalar type.");
				}

				_bitonicSortKernel = this->LoadKernel(KernelPath, KernelName);
			}			

			std::string Name() override { return "Iterated Dispatch Parallel Sort OpenCL"; }
			std::string ShortName() override { return "oclIterDispatch"; }

			bool Process() override
			{
				cl::KernelFunctor<cl::Buffer, int, int, int> bitonicKernelWrapper{ _bitonicSortKernel };
				auto ki = myBase::getKernelInfo(_bitonicSortKernel, this->_device);

				for (int i = 0, start = 0; i < segmentLengths.size(); ++i) {
					assert(start + segmentLengths[i] < _length);

					int stagesNumber = 0;
					for (int temp = segmentLengths[i]; temp > 1; temp >>= 1)
						++stagesNumber;

					size_t globalThreads = segmentLengths[i] >> 1;
					size_t localWorkSize = min(ki.kernelWorkGroupSize, globalThreads);
					auto enqueueArgs = cl::EnqueueArgs{ this->queue, cl::NDRange{globalThreads}, cl::NDRange{ localWorkSize } };

					for (int stage = 0; stage < stagesNumber; ++stage) {
						for (int pass = 0; pass < stage + 1; ++pass) {
							bitonicKernelWrapper(enqueueArgs, _buffer, start, stage, pass);
						}
					}
					
					start += segmentLengths[i];
				}

				return true;
			}			

			bool Write(const SegmentedArray<T>& input) override
			{
				if (_length != input.Data.size()) {
					_length = input.Data.size();
					_buffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(T) * _length, nullptr);
				}
				segmentLengths.resize(input.Counts.size());
				Estd::copy(input.Counts, segmentLengths);

				cl::copy(this->queue, begin(input.Data), end(input.Data), _buffer);

				return true;
			}

			bool Read(SegmentedArray<T>& result) override
			{
				result.Counts.resize(segmentLengths.size());
				Estd::copy(segmentLengths, result.Counts);

				cl::copy(this->queue, _buffer, begin(result.Data), end(result.Data));

				return true;
			}

		};
	}
}