#pragma once

#include "BlackBox.Core/Extensions.h"
#include "BlackBox.Core/ExtendedStd.h"
#include "BlackBox.Core/SegmentedArray.h"
#include "CudaBlackBox.h"
#include "BlellochPlus.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <exception>
#include <algorithm>
#include <cassert>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))

namespace BBox
{
	namespace CUDA
	{

		template <typename T>
		__global__ void bitonicSortParallel (T *data, const int dataLength, int *rank, int *offset, const int rankLength)
		{
			int localId = blockDim.x * blockIdx.x + threadIdx.x;

			if (localId >= rankLength)
				return;

			int length = rank[localId];
			data += offset[localId];

			int stagesNumber = 0;
			for (int temp = length; temp > 1; temp >>= 1)
				++stagesNumber;

			for (int stage = 0; stage < stagesNumber; ++stage) {
				for (int pass = 0; pass < stage + 1; ++pass) {
					int pairDistance = 1 << (stage - pass);

					for (int i = 0; i < (length >> 1); ++i) {

						//uint leftId = (i % pairDistance) + ((i / pairDistance) * blockWidth);
						int leftId = (i & (pairDistance - 1)) | ((i & ~(pairDistance - 1)) << 1);
						int rightId = leftId + pairDistance;

						int leftKey = data[leftId];
						int rightKey = data[rightId];

						int greaterKey;
						int lesserKey;
						if (leftKey > rightKey) {
							greaterKey = leftKey;
							lesserKey = rightKey;
						}
						else {
							greaterKey = rightKey;
							lesserKey = leftKey;
						}

						// sameDirectionBlockWidth = 1 << stage;
						// (i / sameDirectionBlockWidth) % 2 == 1
						int direction = -((i >> stage) & 1);
						data[leftId] = (greaterKey & direction) | (lesserKey & ~direction);
						data[rightId] = (lesserKey & direction) | (greaterKey & ~direction);
					}
				}
			}
		}

		using namespace BBox::Core;

		template <typename T>
		class ParallelDispatch : public CudaBlackBox<SegmentedArray<T>, SegmentedArray<T>>
		{
			//typedef T myType;
			//typedef CudaCLBlackBox<SegmentedArray<T>, std::vector<T>> myBase;

			int _rankLength;

			T* _dataBuffer;
			int* _rankBuffer;
			int* _offsetBuffer;

			BlellochPlus<int> _prefixSum;

			std::vector<int> segmentLengths;

		public:
			// //template <typename T>
			// BBox::CUDA::ParallelDispatch::ParallelDispatch()
			// {
			// 	//if (std::is_scalar<myType>::value) {
			// 	//	declareTypes["__T__"] = typeid(myType).hash_code();
			// 	//}
			// 	//else {
			// 	//	// todo don't know what to do, maybe throw an exception
			// 	//	throw std::runtime_error("Cannot create on non-scalar type.");
			// 	//}
			// }

			std::string Name() override { return "Parallel Dispatch Sort CUDA"; }
			std::string ShortName() override { return "cuParDispatch"; }

			bool Process() override
			{
				int globalThreads = _rankLength;
				int localWorkSize = MIN(this->BlockSize, globalThreads);

				_prefixSum.Process();

				bitonicSortParallel<T><<< static_cast<int>(ceil(static_cast<float>(globalThreads) / localWorkSize)), localWorkSize >>>(
					_dataBuffer, this->_length, _rankBuffer, _offsetBuffer, _rankLength);

				return true;
			}			

			bool Write(const SegmentedArray<T>& input) override
			{
				this->_length = input.Data.size();
				_rankLength = input.Counts.size();

				cudaError_t err = cudaFree(_dataBuffer);
				if (err != cudaSuccess)
					throw err;
				//_dataBuffer = nullptr;
				err = cudaMalloc(&_dataBuffer, this->_length * sizeof(T));
				if (err != cudaSuccess)
					throw err;

				segmentLengths.resize(input.Counts.size());
				Estd::copy(input.Counts, segmentLengths);
				
				_prefixSum.Preprocess(NextGreatestPowerOfTwo(_rankLength));
				_rankBuffer = _prefixSum.inputBuffer();
				_offsetBuffer = _prefixSum.outputBuffer();

				cudaMemcpy(_dataBuffer, &input.Data.front(), this->_length * sizeof(T), cudaMemcpyHostToDevice);
				cudaMemcpy(_rankBuffer, &input.Counts.front(), NextGreatestPowerOfTwo(_rankLength) * sizeof(int), cudaMemcpyHostToDevice);

				return true;
			}

			bool Read(SegmentedArray<T>& result) override
			{
				result.Data.resize(this->_length);
				result.Counts.resize(_rankLength);
				assert(this->_length == result.Data.size());
				cudaMemcpy(&result.Data.front(), _dataBuffer, this->_length * sizeof(T), cudaMemcpyDeviceToHost);
				Estd::copy(segmentLengths, result.Counts);

				return true;
			}
		};
	}
}
