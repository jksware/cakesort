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

#include "BlackBox.Core/Extensions.h"

#include <vector>
#include <cassert>
#include <stdexcept>
#include <cmath>

#include "OpenCLBlackBox.h"

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))

namespace BBox
{
	namespace OpenCL
	{
		using namespace BBox::Core;

		template <typename T>
		class BlellochPlus : public OpenCLBlackBox<std::vector<T>, std::vector<T>>
		{
			const std::string KernelPath = std::string("Kernels") + SEP + "ScanLargeArrays_Kernels.cl";
			const std::string BScanKernelName = "ScanLargeArrays";
			const std::string BAddKernelName = "blockAddition";
			const std::string PScanKernelName = "prefixSum";

			cl::Kernel _bScanKernel;
			cl::Kernel _bAddKernel;
			cl::Kernel _pScanKernel;

			int _blockSize;
			int _passes;

			cl::Buffer _inputBuffer;
			std::vector<cl::Buffer> _outputBuffers;
			std::vector<cl::Buffer> _blockSumBuffers;
			cl::Buffer _tempBuffer;

			typedef T myType;
			typedef OpenCLBlackBox<std::vector<T>, std::vector<T>> myBase;

			using myBase::_length;

		public:
			BlellochPlus() = delete;

			BlellochPlus(cl::Context context, cl::Device device, cl::CommandQueue queue)
				: myBase{ context, device, queue }
			{
				if (std::is_scalar<myType>::value) {
					this->declareTypes["__T__"] = typeid(myType).hash_code();
				}
				else {
					// todo don't know what to do, maybe throw an exception
					throw std::runtime_error("Cannot create on non-scalar type.");
				}

				_bScanKernel = this->LoadKernel(KernelPath, BScanKernelName);
				_bAddKernel = this->LoadKernel(KernelPath, BAddKernelName);
				_pScanKernel = this->LoadKernel(KernelPath, PScanKernelName);
			}

			std::string Name() override { return "Blelloch++ OpenCL Prefix Sum (Scan Large Arrays)"; }
			std::string ShortName() override { return "oclBlelloch++"; }

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
				this->_outputBuffers[0] = outputBuffer;
			}

			inline cl::Buffer outputBuffer() const
			{
				return _outputBuffers[0];
			}			

			bool Process() override
			{
				if (_length == 1) {
					std::vector<int> zero{ 1 };
					zero[0] = { 0 };
					//cudaMemcpy(&_outputBuffers[0], zero, sizeof(int), cudaMemcpyHostToDevice);
					cl::copy(this->queue, zero.begin(), zero.end(), _outputBuffers[0]);
				}
				if (_length < 2) {
					return true;
				}

				BScan(_length, _inputBuffer, _outputBuffers[0], _blockSumBuffers[0]);
				for (int i = 1; i < _passes; i++) {
					BScan(static_cast<int>(_length / std::pow(_blockSize, i)), _blockSumBuffers[i - 1], _outputBuffers[i], _blockSumBuffers[i]);
				}

				int tempLength = static_cast<int>(_length / std::pow(_blockSize, _passes));
				PScan(tempLength, _blockSumBuffers[_passes - 1], _tempBuffer);

				BAddition(static_cast<int>(_length / std::pow(_blockSize, _passes - 1)), _tempBuffer, _outputBuffers[_passes - 1]);
				for (int i = _passes - 1; i > 0; i--) {
					BAddition(static_cast<int>(_length / std::pow(_blockSize, i - 1)), _outputBuffers[i], _outputBuffers[i - 1]);
				}

				return true;
			}			

			void BScan(int len, const cl::Buffer& inputBuffer, const cl::Buffer& outputBuffer, const cl::Buffer& blockSumBuffer)
			{
				size_t globalWorkSize = len / 2;
				size_t localWorkSize = _blockSize / 2;

				if (globalWorkSize * localWorkSize == 0)
					return;

				auto bScanKernelWrapper = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int, cl::Buffer>{ _bScanKernel };

				bScanKernelWrapper(cl::EnqueueArgs{ this->queue, cl::NDRange{ globalWorkSize }, cl::NDRange { localWorkSize } },
					outputBuffer, inputBuffer, cl::Local(_blockSize * sizeof(T)), _blockSize, blockSumBuffer);
			}

			void PScan(int len, const cl::Buffer& inputBuffer, const cl::Buffer& outputBuffer)
			{
				size_t globalWorkSize = len / 2;
				size_t localWorkSize = len / 2;

				if (globalWorkSize * localWorkSize == 0)
					return;

				auto pScanKernelWrapper = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int>{ _pScanKernel };

				pScanKernelWrapper(cl::EnqueueArgs{ this->queue, cl::NDRange{ globalWorkSize }, cl::NDRange{ localWorkSize } },
					outputBuffer, inputBuffer, cl::Local((len + 1) * sizeof(T)), len);
			}

			void BAddition(int len, const cl::Buffer& inputBuffer, const cl::Buffer& outputBuffer)
			{
				size_t globalWorkSize = len;
				size_t localWorkSize = _blockSize;

				if (globalWorkSize * localWorkSize == 0)
					return;

				auto bAddKernelWrapper = cl::KernelFunctor<cl::Buffer, cl::Buffer>{ _bAddKernel };

				bAddKernelWrapper(cl::EnqueueArgs{ this->queue, cl::NDRange{ globalWorkSize }, cl::NDRange{ localWorkSize } },
					inputBuffer, outputBuffer);
			}

			void Preprocess(int length)
			{
				if (!IsPowerOfTwo(length))
					throw std::length_error("Array length must be a power of two.");

				_length = length;

				if (length < 1) {
					_outputBuffers = std::vector<cl::Buffer>(1);
					return;
				}

				_inputBuffer = cl::Buffer(this->context, CL_MEM_READ_ONLY, _length * sizeof(T), nullptr);

				auto bscanKi = myBase::getKernelInfo(_bScanKernel, this->_device);
				auto pscanKi = myBase::getKernelInfo(_pScanKernel, this->_device);
				auto baddKi = myBase::getKernelInfo(_bAddKernel, this->_device);
				size_t temp = MIN(bscanKi.kernelWorkGroupSize, pscanKi.kernelWorkGroupSize);
				temp = MIN(temp, baddKi.kernelWorkGroupSize);

				//_blockSize = MIN(_localSize, _length / 2);
				_blockSize = MAX(2, MIN(temp, _length / 2));

				// Calculate number of passes required
				float t = std::log(static_cast<float>(_length)) / std::log(static_cast<float>(_blockSize));
				_passes = static_cast<int>(t);

				// If t is equal to pass
				if (_passes > 1 && std::abs(t - _passes) < 1e-7f)
					_passes--;

				if (_passes == 0)
					++_passes;

				_outputBuffers = std::vector<cl::Buffer>(_passes);
				for (int i = 0; i < _passes; ++i) {
					int size = static_cast<int>(_length / std::pow(_blockSize, i));
					if (size)
						_outputBuffers[i] = cl::Buffer(this->context, CL_MEM_READ_WRITE, size * sizeof(T), nullptr);
				}

				_blockSumBuffers = std::vector<cl::Buffer>(_passes);
				for (int i = 0; i < _passes; ++i) {
					int size = static_cast<int>(_length / std::pow(_blockSize, i + 1));
					if (size)
						_blockSumBuffers[i] = cl::Buffer(this->context, CL_MEM_READ_WRITE, size * sizeof(T), nullptr);
				}

				int tempLength = static_cast<int>(_length / std::pow(_blockSize, _passes));
				if (tempLength)
					_tempBuffer = cl::Buffer(this->context, CL_MEM_READ_WRITE, tempLength * sizeof(T), nullptr);
			}			

			bool Write(const std::vector<T>& input) override
			{
				Preprocess(input.size());
				cl::copy(this->queue, begin(input), end(input), _inputBuffer);

				return true;
			}			

			bool Read(std::vector<T>& result) override
			{
				result.resize(_length);
				cl::copy(this->queue, _outputBuffers[0], begin(result), end(result));

				return true;
			}
		};
	}
}
