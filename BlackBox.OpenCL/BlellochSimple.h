/**********************************************************************
Copyright ©2014 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#pragma once

#include "OpenCLBlackBox.h"

#include <string>
#include <exception>
#include <cassert>

namespace BBox
{
	namespace OpenCL
	{
		using namespace BBox::Core;

		template <typename T>
		class BlellochSimple : public OpenCLBlackBox<std::vector<T>, std::vector<T>>
		{
			typedef T myType;
			typedef OpenCLBlackBox<std::vector<T>, std::vector<T>> myBase;

			const std::string KernelPath = std::string("Kernels") + SEP + "PrefixSum_Kernels.cl";
			const std::string KernelGroupName = "group_prefixSum";
			const std::string KernelGroupNameExclusive = "group_prefixSumExclusive";
			const std::string KernelGlobalName = "global_prefixSum";
			const std::string KernelGlobalNameExclusive = "global_prefixSumExclusive";

			cl::Kernel _globalKernel;
			cl::Kernel _globalKernelExclusive;
			cl::Kernel _groupKernel;
			cl::Kernel _groupKernelExclusive;

			cl::Buffer _inputBuffer;
			cl::Buffer _outputBuffer;

			size_t kernelWorkGroupSize;
			
			using myBase::_length;

		public:
			BlellochSimple() = delete;

			BlellochSimple(cl::Context context, cl::Device device, cl::CommandQueue queue, bool inclusive = true)
				: myBase{ context, device, queue }, inclusive(inclusive)
			{

				if (std::is_scalar<myType>::value) {
					this->declareTypes["__T__"] = typeid(myType).hash_code();
				}
				else {
					// todo don't know what to do, maybe throw an exception
					throw std::runtime_error("Cannot create on non-scalar type.");
				}

				_globalKernel = this->LoadKernel(KernelPath, KernelGlobalName);
				_globalKernelExclusive = this->LoadKernel(KernelPath, KernelGlobalNameExclusive);
				_groupKernel = this->LoadKernel(KernelPath, KernelGroupName);
				_groupKernelExclusive = this->LoadKernel(KernelPath, KernelGroupNameExclusive);
			}

			bool inclusive = 0;

			inline void inputBuffer(const cl::Buffer& inputBuffer)
			{
				_inputBuffer = inputBuffer;
			}

			inline cl::Buffer inputBuffer() const
			{
				return _inputBuffer;
			}

			inline void outputBuffer(const cl::Buffer& outputBuffer)
			{
				_outputBuffer = outputBuffer;
			}

			inline cl::Buffer outputBuffer() const
			{
				return _outputBuffer;
			}

			std::string Name() override { return "Blelloch OpenCL Prefix Sum"; }
			std::string ShortName() override { return "oclBlelloch"; }

			bool Process() override
			{
				auto ki = myBase::getKernelInfo(_groupKernel, this->_device);
				kernelWorkGroupSize = ki.kernelWorkGroupSize;

				size_t localThreads = kernelWorkGroupSize;
				size_t localDataSize = localThreads << 1;   // Each thread work on 2 elements

				cl_ulong availableLocalMemory = ki.localMemSize - ki.localMemoryUsed;
				cl_ulong neededLocalMemory = localDataSize * sizeof(T);

				if (neededLocalMemory > availableLocalMemory)
					throw std::length_error("Unsupported: Insufficient local memory on device.");

				/*
				runGroupKernel(1);
				*/
				for (size_t stride = 1; stride < _length; stride *= localDataSize) {
					// Need atlest 2 elements to process the kernel
					if ((_length / stride) > 1)
						runGroupKernel(stride);

					// Call global_kernel to update all elements
					if (stride > 1)
						runGlobalKernel(stride);
				}
				return true;
			}

			inline void runGroupKernel(size_t stride)
			{
				size_t dataSize = _length / stride;
				size_t localThreads = kernelWorkGroupSize;
				size_t globalThreads = (dataSize + 1) / 2;    // Actual threads needed
															// Set global thread size multiple of local thread size.
				globalThreads = ((globalThreads + localThreads - 1) / localThreads) * localThreads;

				cl::Event eventOut;
				cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int, int> _groupKernelWrapper { 
					inclusive ? _groupKernel : _groupKernelExclusive };

				auto waitEvent = _groupKernelWrapper(
					cl::EnqueueArgs{ this->queue, cl::NDRange{ globalThreads }, cl::NDRange{ localThreads} },
					_outputBuffer, 
					stride > 1 ? _outputBuffer : _inputBuffer,
					cl::Local(2 * localThreads * sizeof(T)),
					_length,
					stride);

				this->queue.flush();
				waitEvent.wait();
			}

			inline void runGlobalKernel(size_t stride)
			{
				size_t localThreads = kernelWorkGroupSize;
				size_t localDataSize = localThreads << 1;   // Each thread works on 2 elements

															// Set number of threads needed for global_kernel.
				size_t globalThreads = _length - stride;
				globalThreads -= (globalThreads / (stride * localDataSize)) * stride;

				// Set global thread size multiple of local thread size.
				globalThreads = ((globalThreads + localThreads - 1) / localThreads) * localThreads;

				cl::KernelFunctor<cl::Buffer, int, int> _globalKernelWrapper {
					inclusive ? _globalKernel : _globalKernelExclusive };

				auto waitEvent = _globalKernelWrapper(
					cl::EnqueueArgs{ this->queue, cl::NDRange { globalThreads }, cl::NDRange { localThreads } },
					_outputBuffer, stride, _length);

				this->queue.flush();
				waitEvent.wait();
			}			

			bool Write(const std::vector<T>& input) override
			{
				if (_length != input.size()) {
					_length = input.size();
					this->_inputBuffer = cl::Buffer(this->context, CL_MEM_READ_ONLY, _length * sizeof(T), nullptr);
					this->_outputBuffer = cl::Buffer(this->context, CL_MEM_WRITE_ONLY, _length * sizeof(T), nullptr);
				}
				cl::copy(this->queue, begin(input), end(input), _inputBuffer);
				return true;
			}			

			bool Read(std::vector<T>& result) override
			{
				assert(result.size() == _length);
				cl::copy(this->queue, _outputBuffer, begin(result), end(result));

				return true;
			}		
		};
	}
}