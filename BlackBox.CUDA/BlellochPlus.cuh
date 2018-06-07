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

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include "BlackBox.Core/Extensions.h"
#include "CudaBlackBox.h"

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))

namespace BBox
{
	namespace CUDA
	{
		/*
		* ScanLargeArrays : Scan is done for each block and the sum of each
		* block is stored in separate array (sumBuffer). SumBuffer is scanned
		* and results are added to every value of next corresponding block to
		* compute the scan of a large array.(not limited to 2*MAX_GROUP_SIZE)
		* Scan uses a balanced tree algorithm. See Belloch, 1990 "Prefix Sums
		* and Their Applications"
		* @param output output uint
		* @param input  input uint
		* @param block  local memory used in the kernel
		* @param sumBuffer  sum of blocks
		* @param length length of the input uint
		*/

		template <typename T>
		__global__ void BAddDev(T* input, T* output)
		{
			int localId = threadIdx.x;//get_local_id(0);
			int globalId = blockDim.x * blockIdx.x + threadIdx.x; //get_global_id(0);
			int groupId = blockIdx.x;//get_group_id(0);

			__shared__ T value[1];

			/* Only 1 thread of a group will read from global buffer */
			if (localId == 0) {
				value[0] = input[groupId];
			}
			__syncthreads();

			output[globalId] += value[0];
		}

		template <typename T>
		__global__ void BScanDev(T *output, T *input, const int block_size, T *sumBuffer)
		{
			extern __shared__ T block[];

			int tid = threadIdx.x;//get_local_id(0);
			int gid = blockDim.x * blockIdx.x + threadIdx.x; //get_global_id(0);
			int bid = blockIdx.x;//get_group_id(0);

			/* Cache the computational window in shared memory */
			block[2 * tid] = input[2 * gid];
			block[2 * tid + 1] = input[2 * gid + 1];
			__syncthreads();

			T cache0 = block[0];
			T cache1 = cache0 + block[1];

			/* build the sum in place up the tree */
			for (int stride = 1; stride < block_size; stride *= 2) {
				if (2 * tid >= stride) {
					cache0 = block[2 * tid - stride] + block[2 * tid];
					cache1 = block[2 * tid + 1 - stride] + block[2 * tid + 1];
				}
				__syncthreads();
				
				block[2 * tid] = cache0;
				block[2 * tid + 1] = cache1;

				__syncthreads();
			}

			if (tid == 0)
				/* store the value in sum buffer before making it to 0 */
				sumBuffer[bid] = block[block_size - 1];

			/*write the results back to global memory */
			if (tid == 0) {
				output[2 * gid] = 0;
				output[2 * gid + 1] = block[2 * tid];
			}
			else {
				output[2 * gid] = block[2 * tid - 1];
				output[2 * gid + 1] = block[2 * tid];
			}

		}

		template <typename T>
		__global__ void PScanDev(T *output, T *input, const int block_size)
		{
			extern __shared__ T block[];

			int tid = threadIdx.x;//get_local_id(0);
			int gid = blockDim.x * blockIdx.x + threadIdx.x; //get_global_id(0);
			//int bid = blockIdx.x;//get_group_id(0);

			/* Cache the computational window in shared memory */
			block[2 * tid] = input[2 * gid];
			block[2 * tid + 1] = input[2 * gid + 1];
			__syncthreads();
			
			T cache0 = block[0];
			T cache1 = cache0 + block[1];

			/* build the sum in place up the tree */
			for (int stride = 1; stride < block_size; stride *= 2) {
				if (2 * tid >= stride) {
					cache0 = block[2 * tid - stride] + block[2 * tid];
					cache1 = block[2 * tid + 1 - stride] + block[2 * tid + 1];
				}
				//barrier(CLK_LOCAL_MEM_FENCE);
				__syncthreads();

				block[2 * tid] = cache0;
				block[2 * tid + 1] = cache1;

				//barrier(CLK_LOCAL_MEM_FENCE);
				__syncthreads();
			}

			/*write the results back to global memory */
			if (tid == 0) {
				output[2 * gid] = 0;
				output[2 * gid + 1] = block[2 * tid];
			}
			else {
				output[2 * gid] = block[2 * tid - 1];
				output[2 * gid + 1] = block[2 * tid];
			}
		}

		using namespace BBox::Core;

		template <typename T>
		class BlellochPlus : public CudaBlackBox<std::vector<T>, std::vector<T>>
		{
			int _blockSize;
			int _passes;

			T *_inputBuffer;
			std::vector<T*> _outputBuffers;
			std::vector<T*> _blockSumBuffers;
			T *_tempBuffer;

			//typedef T myType;
			//typedef OpenCLBlackBox<std::vector<T>, std::vector<T>> myBase;

		public:
			std::string Name() override { return "Blelloch++ CUDA Prefix Sum (Scan Large Arrays)"; }
			std::string ShortName() override { return "cuBlelloch++"; }

			inline int* & inputBuffer()
			{
				return _inputBuffer;
			}

			inline int* & outputBuffer()
			{
				return _outputBuffers[0];
			}

			bool Process() override
			{
				if (this->_length == 1) {
					int zero[1] = { 0 };
					cudaMemcpy(&_outputBuffers[0], zero, sizeof(T), cudaMemcpyHostToDevice);
				}
				if (this->_length < 2) {
					return true;
				}
				
				BScan(this->_length, _inputBuffer, _outputBuffers[0], _blockSumBuffers[0]);
				for (int i = 1; i < _passes; i++) {
					BScan(static_cast<int>(this->_length / std::pow(_blockSize, i)),
						_blockSumBuffers[i - 1], _outputBuffers[i], _blockSumBuffers[i]);
				}

				int tempLength = static_cast<int>(this->_length / std::pow(this->_blockSize, _passes));
				PScan(tempLength, _blockSumBuffers[_passes - 1], _tempBuffer);

				BAddition(static_cast<int>(this->_length / std::pow(this->_blockSize, _passes - 1)),
					_tempBuffer, _outputBuffers[_passes - 1]);
				for (int i = _passes - 1; i > 0; i--) {
					BAddition(static_cast<int>(this->_length / std::pow(_blockSize, i - 1)),
						_outputBuffers[i], _outputBuffers[i - 1]);
				}

				return true;
			}

			void BScan(int len, T* const& inputBuffer, T* const& outputBuffer, T* const& blockSumBuffer)
			{
				int globalWorkSize = len / 2;
				int localWorkSize = _blockSize / 2;

				if (globalWorkSize * localWorkSize == 0)
					return;

				BScanDev<T><<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/localWorkSize)), 
					localWorkSize, _blockSize * sizeof(T)>>>(
					outputBuffer, inputBuffer, _blockSize, blockSumBuffer);
			}

			void PScan(int len, T* const& inputBuffer, T* const& outputBuffer)
			{ 
				int globalWorkSize = len / 2;
				int localWorkSize = len / 2;

				if (globalWorkSize * localWorkSize == 0)
					return;

				PScanDev<T><<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/localWorkSize)),
					localWorkSize, (len + 1) * sizeof(T)>>>(
					outputBuffer, inputBuffer, len);
			}			

			void BAddition(int len, T* const& inputBuffer, T* const& outputBuffer)
			{
				int globalWorkSize = len;
				int localWorkSize = _blockSize;

				if (globalWorkSize * localWorkSize == 0)
					return;

				BAddDev<T><<<static_cast<int>(ceil(static_cast<float>(globalWorkSize)/localWorkSize)),
				localWorkSize>>>(inputBuffer, outputBuffer);
			}

			void Preprocess(int length)
			{
				if (!IsPowerOfTwo(length))
					throw std::length_error("Array length must be a power of two.");

				this->_length = length;

				if (length < 1) {
					_outputBuffers = std::vector<T*>(1);
					return;
				}

				cudaError_t err = cudaFree(_inputBuffer);
				if (err != cudaSuccess)
					throw err;
				err = cudaMalloc(&_inputBuffer, this->_length * sizeof(T));

				this->_blockSize = MAX(2, MIN(this->BlockSize, this->_length / 2));

				// Calculate number of passes required
				float t = std::log(static_cast<float>(this->_length)) / std::log(static_cast<float>(this->_blockSize));
				_passes = static_cast<int>(t);

				// If t is equal to pass
				if (_passes > 1 && std::abs(t - _passes) < 1e-7f)
					_passes--;

				if (_passes == 0)
					++_passes;

				for (int i = 0; i < _outputBuffers.size(); ++i) {
					err = cudaFree(_outputBuffers[i]);	 
					if (err != cudaSuccess)
						throw err;
				}
				_outputBuffers = std::vector<T*>(_passes);
				for (int i = 0; i < _passes; ++i) {
					int size = static_cast<int>(this->_length / std::pow(this->_blockSize, i));
					err = cudaMalloc(&_outputBuffers[i], size * sizeof(T));
					if (err != cudaSuccess)
						throw err;
				}

				for (int i = 0; i < _blockSumBuffers.size(); ++i) {
					err = cudaFree(_blockSumBuffers[i]);
					if (err != cudaSuccess)
						throw err;
				}
				_blockSumBuffers = std::vector<T*>(_passes);
				for (int i = 0; i < _passes; ++i) {
					int size = static_cast<int>(this->_length / std::pow(this->_blockSize, i + 1));
					err = cudaMalloc(&_blockSumBuffers[i], size * sizeof(T));
					if (err != cudaSuccess)
						throw err;
				}

				int tempLength = static_cast<int>(this->_length / std::pow(this->_blockSize, _passes));
				err = cudaFree(_tempBuffer);
				if (err != cudaSuccess)
					throw err;
				err = cudaMalloc(&_tempBuffer, tempLength * sizeof(T));
				if (err != cudaSuccess)
					throw err;
			}			

			bool Write(const std::vector<T>& input) override
			{
				Preprocess(input.size());
				cudaMemcpy(_inputBuffer, &input.front(), this->_length * sizeof(T), 
					cudaMemcpyHostToDevice);

				return true;
			}			

			bool Read(std::vector<T>& result) override
			{
				result.resize(this->_length);
				cudaMemcpy(&result.front(), _outputBuffers[0], this->_length * sizeof(T), 
					cudaMemcpyDeviceToHost);

				return true;
			}
		};
	}
}
