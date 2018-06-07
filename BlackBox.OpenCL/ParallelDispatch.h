#pragma once

#include "OpenCLBlackBox.h"
#include "BlackBox.Core/SegmentedArray.h"
#include "BlackBox.Core/ExtendedStd.h"
#include "BlackBox.Core/Extensions.h"
#include "BlellochPlus.h"

#include <string>
#include <exception>

namespace BBox
{
	namespace OpenCL
	{
		using namespace BBox::Core;

		template <typename T>
		class ParallelDispatch : public OpenCLBlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			typedef T myType;
			typedef OpenCLBlackBox<SegmentedArray<T>, SegmentedArray<T>> myBase;

			const std::string KernelPath = std::string("Kernels") + SEP + "ParallelBitonicSort_Kernels.cl";
			const std::string KernelName = "bitonicSort";

			int _rankLength;

			cl::Kernel _bitonicSortKernel;

			cl::Buffer _dataBuffer;
			cl::Buffer _rankBuffer;
			cl::Buffer _offsetBuffer;

			BlellochPlus<int> _prefixSum;

			size_t kernelWorkGroupSize;

			std::vector<int> segmentLengths;

			using myBase::_length;

		public:
			ParallelDispatch() = delete;

			ParallelDispatch(cl::Context context, cl::Device device, cl::CommandQueue queue)
				: myBase{ context, device, queue }, _prefixSum{ context, device, queue }
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

			std::string Name() override { return "Parallel Dispatch Sort OpenCL"; }
			std::string ShortName() override { return "oclParDispatch"; }

			bool Process() override
			{
				cl::KernelFunctor<cl::Buffer, int, cl::Buffer, cl::Buffer, int> bitonicParallelKernelWrapper{ _bitonicSortKernel };
				auto ki = myBase::getKernelInfo(_bitonicSortKernel, this->_device);

				size_t globalThreads = _rankLength;
				size_t localWorkSize = min(ki.kernelWorkGroupSize, globalThreads);

				_prefixSum.Process();

				bitonicParallelKernelWrapper(cl::EnqueueArgs{ this->queue, cl::NDRange{globalThreads}, cl::NDRange{ localWorkSize } },
					_dataBuffer, _length, _rankBuffer, _offsetBuffer, _rankLength);

				return true;
			}			

			bool Write(const SegmentedArray<T>& input) override
			{
				if (_length != input.Data.size() || _rankLength != input.Counts.size()) {
					_length = input.Data.size();
					_rankLength = input.Counts.size();

					_dataBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(T) * _length, nullptr);
					_rankBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(T) * NextGreatestPowerOfTwo(_rankLength), nullptr);
					_offsetBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, sizeof(int) * NextGreatestPowerOfTwo(_rankLength), nullptr);
				}
				_prefixSum.Preprocess(NextGreatestPowerOfTwo(_rankLength));
				_prefixSum.inputBuffer(_rankBuffer);
				_prefixSum.outputBuffer(_offsetBuffer);

				cl::copy(this->queue, begin(input.Data), end(input.Data), _dataBuffer);
				cl::copy(this->queue, begin(input.Counts), end(input.Counts), _rankBuffer);

				return true;
			}			

			bool Read(SegmentedArray<T>& result) override
			{
				result.Data.resize(_length);
				cl::copy(this->queue, _dataBuffer, begin(result.Data), end(result.Data));

				result.Counts.resize(_rankLength);
				cl::copy(this->queue, _rankBuffer, begin(result.Counts), end(result.Counts));

				return true;
			}
			
		};
	}
}